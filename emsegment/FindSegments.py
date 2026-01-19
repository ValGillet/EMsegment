import daisy
import json
import logging
import multiprocessing as mp
import numpy as np
import os
import sys
import time

from datetime import date
from funlib.segment.graphs.impl import connected_components
from funlib.persistence import open_ds
from funlib.geometry import Roi, Coordinate
from multiprocessing import Manager
from pymongo import MongoClient

logging.basicConfig(level = logging.INFO)



def find_segments(
                  db_name,
                  fragments_path,
                  edges_collection,
                  thresholds_minmax,
                  thresholds_step,
                  chunk_voxel_size=[100,500,500],
                  num_workers=1,
                  db_host=None,
                  fragments_dataset='frags',
                  chunk_bbox=[],
                  run_type=None,
                  **kwargs):
    '''
    Extract final segments from the region adjacency graph at multiple thresholds.
    Based on https://github.com/funkelab/lsd/blob/master/lsd/tutorial/scripts/04_find_segments.py

    Parameters
    ----------
    db_name : str
        MongoDB database name containing the edge collection.
    fragments_path : str
        Path to zarr container with fragments dataset.
    edges_collection : str
        MongoDB collection name containing RAG edges (e.g., 'edges_hist_quant_25').
    thresholds_minmax : list of float
        [min, max] threshold range for agglomeration.
    thresholds_step : float
        Step size between thresholds in the range.
    chunk_voxel_size : list of int, optional
        Block size in voxels [z, y, x] for reading graph chunks. Default: [100, 500, 500]
    num_workers : int, optional
        Number of parallel workers for reading graph chunks. Default: 1
    db_host : str or None, optional
        MongoDB connection URI. If None, uses localhost. Default: None
    fragments_dataset : str, optional
        Name of dataset containing fragments. Default: 'frags'
    chunk_bbox : list, optional
        Bounding box for processing subset of volume as [start_chunk, end_chunk] indices.
        If empty, processes entire volume. Default: []
    run_type : str or None, optional
        Optional label for organizing output LUTs in subdirectories. Default: None
    **kwargs
        Additional keyword arguments (ignored).

    Returns
    -------
    bool
        True if successful, raises exception otherwise.

    Notes
    -----
    - Reads RAG in parallel blocks using daisy, aggregates into single graph
    - For each threshold, computes connected components to merge fragments into segments
    - LUTs saved as compressed numpy arrays: fragment_segment_lut[fragment_id] = segment_id
    - Output directory: <fragments_path>/luts/fragment_segment_<edges_collection>/[run_type]/
    - LUT filename format: seg_<edges_collection>_<threshold*100>.npz
    - Metadata written to 'info_segmentation' collection in MongoDB

    Examples
    --------
    Extract segments at thresholds from 0.0 to 1.0 with 0.1 steps:

    >>> find_segments(
    ...     db_name='my_agglomeration',
    ...     fragments_path='/data/predictions.zarr',
    ...     edges_collection='edges_hist_quant_25',
    ...     thresholds_minmax=[0.0, 1.0],
    ...     thresholds_step=0.1,
    ...     chunk_voxel_size=[100, 500, 500],
    ...     num_workers=8
    ... )

    Process subset of volume:

    >>> find_segments(
    ...     db_name='my_agglomeration',
    ...     fragments_path='/data/predictions.zarr',
    ...     edges_collection='edges_hist_quant_25',
    ...     thresholds_minmax=[0.0, 0.5],
    ...     thresholds_step=0.05,
    ...     chunk_bbox=[[0, 0, 0], [10, 10, 10]],
    ...     run_type='test_region'
    ... )

    See Also
    --------
    get_connected_components : Compute connected components at single threshold
    read_chunk_graph : Read RAG edges for a block from MongoDB
    '''

    start = time.time()

    logging.info(f'Reading graph from DB: {db_name} and collection: {edges_collection}')
    
    fragments = open_ds(os.path.join(fragments_path, fragments_dataset))
    chunk_size = Coordinate(chunk_voxel_size) * fragments.voxel_size

    # Open RAG DB    
    if len(chunk_bbox) > 0:
        roi_offset = fragments.roi.get_begin() + Coordinate(chunk_bbox[0])*Coordinate(chunk_size)
        roi_size = Coordinate(chunk_size)*(Coordinate(chunk_bbox[1])-Coordinate(chunk_bbox[0]))
        roi = Roi(roi_offset, roi_size)
    else:
        roi = fragments.roi
    
    logging.info(f'Looking for segments in {roi}')
    with Manager() as manager:
        shared_list = manager.list()
        tasks = daisy.Task(
                    task_id = f'Find segments - {db_name}',
                    total_roi = fragments.roi,
                    read_roi = Roi((0,0,0), chunk_size),
                    write_roi = Roi((0,0,0), chunk_size),
                    process_function = lambda b: read_chunk_graph(b, 
                                                                fragments, 
                                                                db_host, 
                                                                db_name, 
                                                                edges_collection,
                                                                shared_list),
                    num_workers = num_workers,
                    read_write_conflict = False,
                    fit = 'shrink'
                        )
        daisy.run_blockwise([tasks])

        shared_list = list(shared_list)

        nodes = np.concatenate([l[0] for l in shared_list])
        edges = np.concatenate([l[1] for l in shared_list])
        scores = np.concatenate([l[2] for l in shared_list])

    logging.info(f'Complete RAG contains {len(nodes)} nodes, {len(edges)} edges')

    out_dir = os.path.join(
        fragments_path,
        'luts',
        f'fragment_segment_{edges_collection}')

    if run_type is not None:
        out_dir = os.path.join(out_dir, run_type)

    os.makedirs(out_dir, exist_ok=True)

    thresholds = [round(i,2) for i in np.arange(
        float(thresholds_minmax[0]),
        float(thresholds_minmax[1])+thresholds_step,
        thresholds_step)]

    start = time.time()

    # Extract connected components per threshold
    try:
        for threshold in thresholds:

            get_connected_components(
                    nodes,
                    edges,
                    scores,
                    threshold,
                    edges_collection,
                    out_dir)

            logging.info(f'Created and stored lookup tables in {time.time() - start}')
        db = MongoClient(db_host)[db_name]
        doc = {
            'task': 'find_segments',
            'date': date.today().strftime('%d%m%Y'),
            'voxel_size': list(fragments.voxel_size),
            'fragments_path': fragments_path,
            'edges_collection': edges_collection,
            'thresholds_minmax': thresholds_minmax,
            'thresholds_step': thresholds_step,
            'chunk_voxel_size': chunk_voxel_size,
            'num_workers': num_workers,
            'fragments_dataset': fragments_dataset,
            'chunk_bbox': chunk_bbox,
            'run_type': run_type
            }
        db['info_segmentation'].insert_one(doc)
        
        return True

    except Exception as e:
        raise(e)
    

def get_connected_components(
        nodes,
        edges,
        scores,
        threshold,
        edges_collection,
        out_dir,
        **kwargs):
    '''
    Compute connected components at a single threshold and save lookup table.

    Parameters
    ----------
    nodes : numpy.ndarray
        1D array of fragment IDs (uint64).
    edges : numpy.ndarray
        2D array of shape (n_edges, 2) containing fragment pairs (u, v) as uint64.
    scores : numpy.ndarray
        1D array of merge scores for each edge (float32).
    threshold : float
        Agglomeration threshold. Edges with scores <= threshold are merged.
    edges_collection : str
        Edge collection name, used in output filename.
    out_dir : str
        Output directory for saving LUT.
    **kwargs
        Additional keyword arguments (ignored).

    Notes
    -----
    - Output LUT maps fragment_id -> segment_id
    - Saved as compressed numpy array: <out_dir>/seg_<edges_collection>_<threshold*100>.npz
    - Lower thresholds produce more segments (less merging)
    - Higher thresholds produce fewer segments (more aggressive merging)

    See Also
    --------
    find_segments : Main function that calls this for multiple thresholds
    '''

    logging.info(f'Getting CCs for threshold {threshold}...')
    components = connected_components(nodes, edges, scores, threshold)
    
    logging.info(f'Creating fragment-segment LUT for threshold {threshold}...')
    lut = np.array([nodes, components])

    logging.info(f'Storing fragment-segment LUT for threshold {threshold}...')

    lookup = f'seg_{edges_collection}_{int(threshold*100)}'

    out_file = os.path.join(out_dir, lookup)

    np.savez_compressed(out_file, fragment_segment_lut=lut)


def read_chunk_graph(block, fragments, db_host, db_name, edges_collection, shared_list):
    '''
    Read RAG edges for a block from MongoDB and append to shared list.

    Called by daisy workers. Queries MongoDB for all edges connected to fragments in the
    current block ROI. Extracts unique fragment IDs, edge pairs, and merge scores, then
    appends to a multiprocessing-shared list for aggregation.

    Parameters
    ----------
    block : daisy.Block
        Block object containing read_roi for the current chunk.
    fragments : funlib.persistence.Array
        Array handle for fragments dataset.
    db_host : str
        MongoDB connection URI.
    db_name : str
        MongoDB database name.
    edges_collection : str
        MongoDB collection name containing edges.
    shared_list : multiprocessing.Manager.list
        Shared list for accumulating (nodes, edges, scores) tuples across workers.

    Notes
    -----
    - Queries edges where either u or v is in the block's fragment set
    - Returns tuples of (nodes, edges, scores) for parallel aggregation
    - Edges stored as documents: {'u': fragment_id, 'v': fragment_id, 'merge_score': float}
    - Uses $or query to find all edges touching fragments in the block
    '''

    client = MongoClient(db_host)
    edges_coll = client[db_name][edges_collection]
    
    data = fragments[block.read_roi]

    nodes = np.unique(data)

    edges = list(edges_coll.find({'$or': [{'u':{'$in': nodes.astype(int).tolist()}}, 
                                            {'v':{'$in': nodes.astype(int).tolist()}}]},
                                    {'_id': 0, 
                                        'u': 1,
                                        'v': 1,
                                        'merge_score': 1}))

    scores = np.array([e['merge_score'] for e in edges], dtype=np.float32)
    edges = np.array([[e['u'], e['v']] for e in edges], dtype=np.uint64)

    shared_list.append((nodes, edges, scores))


if __name__ == '__main__':

    config_file = sys.argv[1]

    with open(config_file, 'r') as f:
        config = json.load(f)

    start = time.time()
    find_segments(**config)
   
    edges_collection = config['edges_collection']

    logging.info(f'Found segments for {edges_collection}')
    logging.info(f'Took {time.time() - start} seconds to find segments and store LUTs')