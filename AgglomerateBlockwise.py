import daisy
import hashlib
import json
import logging
import os
import pymongo
import sys
import time

from datetime import date
from funlib.persistence import open_ds
from funlib.geometry import Roi, Coordinate

from emsegment.utils.block_wise_process import check_block, daisy_call

logging.basicConfig(level=logging.INFO)
logging.getLogger('pymongo').setLevel(logging.WARNING) # Hide pymongo output when debugging


def agglomerate_blockwise(
                pred_path,
                chunk_voxel_size,
                context_px,
                db_name,
                merge_function,
                num_workers,
                db_host=None,
                pred_dataset='pred_affs',
                fragments_path=None,
                fragments_dataset='frags',
                edges_collection='edges',
                threshold=10,
                lsd_sigma=60.0
               ):
    '''
    Build a region adjacency graph (RAG) and compute edge scores for fragment agglomeration.
    Based on https://github.com/funkelab/lsd/blob/master/lsd/tutorial/scripts/03_agglomerate_blockwise.py

    Parameters
    ----------
    pred_path : str
        Path to zarr container with prediction data (affinities or LSDs).
    chunk_voxel_size : list of int
        Block size in voxels [z, y, x] for processing.
    context_px : list of int
        Context (overlap) size in voxels [z, y, x] for block boundaries.
    db_name : str
        MongoDB database name for tracking progress and storing edges.
    merge_function : str
        Edge scoring function name (e.g., 'hist_quant_25', 'mean'). Determines how
        prediction values are aggregated across fragment boundaries.
    num_workers : int
        Number of CPU worker processes for parallel RAG construction.
    db_host : str or None, optional
        MongoDB connection URI. If None, uses localhost. Default: None
    pred_dataset : str, optional
        Name of dataset in pred_path containing predictions. Auto-detects mode:
        3 channels = affinities, 10 channels = LSDs. Default: 'pred_affs'
    fragments_path : str or None, optional
        Path to zarr with fragments. If None, uses pred_path. Default: None
    fragments_dataset : str, optional
        Name of dataset containing fragments. Default: 'frags'
    edges_collection : str, optional
        Base name for MongoDB edge collection. Will be suffixed with merge_function
        (e.g., 'edges_hist_quant_25') or '_lsd' for LSD mode. Default: 'edges'
    threshold : float, optional
        Agglomeration threshold for reference (stored in metadata, not used during
        RAG construction). Default: 10
    lsd_sigma : float, optional
        Sigma for Gaussian smoothing when using LSD mode (in nm). Default: 60.0

    Returns
    -------
    bool
        True if all blocks completed successfully, False otherwise.

    Notes
    -----
    - Automatically detects mode from prediction shape: 3 channels=affs, 10 channels=LSDs
    - Edge collection name includes merge function for tracking different scoring methods
    - Edges stored as MongoDB documents with fragment IDs (u, v) and merge score
    - MongoDB collection 'blocks_agglomerated_<merge_function>' tracks completion
    - Metadata written to 'info_segmentation' collection upon successful completion
    - Workers write logs to: workers/tmp_extract_fragments_blockwise/agglomerate_blockwise_<id>.{out,err}

    Examples
    --------
    Build RAG using histogram quantile merge function:

    >>> agglomerate_blockwise(
    ...     pred_path='/data/predictions.zarr',
    ...     chunk_voxel_size=[100, 500, 500],
    ...     context_px=[10, 50, 50],
    ...     db_name='my_agglomeration',
    ...     merge_function='hist_quant_25',
    ...     num_workers=8,
    ...     pred_dataset='pred_affs',
    ...     fragments_dataset='frags',
    ...     edges_collection='edges'
    ... )

    Build RAG with LSD predictions:

    >>> agglomerate_blockwise(
    ...     pred_path='/data/predictions.zarr',
    ...     chunk_voxel_size=[100, 500, 500],
    ...     context_px=[10, 50, 50],
    ...     db_name='my_agglomeration',
    ...     merge_function='mean',
    ...     num_workers=8,
    ...     pred_dataset='pred_lsds',
    ...     lsd_sigma=60.0
    ... )

    See Also
    --------
    workers.AgglomerateWorker.agglomerate_worker : Worker that builds RAG subgraphs
    FindSegments.find_segments : Extract final segments from RAG at threshold
    start_worker : Spawns worker subprocesses
    '''

    fragments_path = pred_path if fragments_path is None else fragments_path

    logging.info(f'Reading predictions from {pred_path}')
    logging.info(f'Reading fragments from {fragments_path}')
    pred = open_ds(os.path.join(pred_path, pred_dataset), mode = 'r')
    fragments = open_ds(os.path.join(fragments_path, fragments_dataset), mode = 'r')

    if pred.shape[0] == 3:
        mode = 'affs'
        edges_collection = f'{edges_collection}_{merge_function}'
        blocks_collection_name = 'blocks_agglomerated_' + merge_function
    elif pred.shape[0] == 10:
        mode = 'lsds'
        edges_collection = f'{edges_collection}_lsd'
        blocks_collection_name = 'blocks_agglomerated_lsd'
    else:
        raise ValueError(f'Unexpected shape for prediction dataset: {pred.shape}')

    # Prepare variables
    chunk_size = Coordinate(chunk_voxel_size) * pred.voxel_size
    context = Coordinate(context_px) * pred.voxel_size
    total_roi = fragments.roi

    read_roi = Roi((0,0,0), chunk_size).grow(context, context)
    write_roi = Roi((0,0,0), chunk_size)

    # Prepare MongoDB to log blocks
    client = pymongo.MongoClient(db_host)
    db = client[db_name]
    if blocks_collection_name not in db.list_collection_names():
        blocks_agglomerated = db[blocks_collection_name]
        blocks_agglomerated.create_index(
                                        [('block_id', pymongo.ASCENDING)],
                                        name = 'block_id')

    # Blockwise watershed
    tasks = daisy.Task(
                    task_id=f'Agglomerate-{db_name}',
                    total_roi=total_roi,
                    read_roi=read_roi,
                    write_roi=write_roi,
                    process_function=lambda: start_worker(
                                                        pred_path,
                                                        pred_dataset,
                                                        fragments_path,
                                                        fragments_dataset,
                                                        db_host,
                                                        db_name,
                                                        edges_collection,
                                                        merge_function,
                                                        threshold,
                                                        lsd_sigma,
                                                        mode
                                                        ),
                    check_function=lambda b: check_block(
                                        b, 
                                        db_host, db_name,
                                        blocks_collection_name),
                    num_workers = num_workers,
                    read_write_conflict = False,
                    fit = 'shrink')
    
    done = daisy.run_blockwise([tasks])
    
    if done:
        doc = {
            'task': 'agglomeration',
            'date': date.today().strftime('%d%m%Y'),
            'voxel_size': list(pred.voxel_size),
            'size_roi_nm': list(total_roi.get_shape()),
            'start_roi_nm': list(total_roi.begin),
            'pred_path': pred_path,
            'pred_dataset': pred_dataset,
            'fragments_path': fragments_path,
            'fragments_dataset': fragments_dataset,
            'chunk_voxel_size': chunk_voxel_size,
            'context_px': context_px,
            'merge_function': merge_function,
            'edges_collection': edges_collection,
            'threshold': threshold,
            'lsd_sigma': lsd_sigma,
            'mode': mode
            }
        db['info_segmentation'].insert_one(doc)
    
    return done

def start_worker(
                 pred_path,
                 pred_dataset,
                 fragments_path,
                 fragments_dataset,
                 db_host,
                 db_name,
                 edges_collection,
                 merge_function,
                 threshold,
                 lsd_sigma,
                 mode):
    '''
    Spawn an AgglomerateWorker subprocess for block-wise RAG construction.
    Called by daisy for each worker process. 

    Parameters
    ----------
    pred_path : str
        Path to zarr container with predictions.
    pred_dataset : str
        Name of dataset in pred_path containing predictions.
    fragments_path : str
        Path to zarr container with fragments.
    fragments_dataset : str
        Name of dataset containing fragments.
    db_host : str
        MongoDB connection URI.
    db_name : str
        MongoDB database name for storing edges and tracking progress.
    edges_collection : str
        MongoDB collection name for storing edges (includes merge function suffix).
    merge_function : str
        Edge scoring function name (e.g., 'hist_quant_25', 'mean').
    threshold : float
        Agglomeration threshold for agglomeration with affinities.
    lsd_sigma : float
        Sigma for Gaussian smoothing when using LSD mode (in nm).
    mode : str
        Processing mode, either 'affs' or 'lsds'.

    Notes
    -----
    - Worker ID is obtained from daisy.Context.from_env()
    - Worker logs written to: workers/tmp_extract_fragments_blockwise/agglomerate_blockwise_<id>.{out,err}
    - Configuration file persists for debugging

    See Also
    --------
    workers.AgglomerateWorker.agglomerate_worker : Worker subprocess implementation
    '''
    
    daisy_context = daisy.Context.from_env()
    worker_id = int(daisy_context.get('worker_id'))
    logging.info(f'Worker {worker_id} started...')

    worker_script = os.path.join(os.path.dirname(__file__), 'workers', 'AgglomerateWorker.py')

    output_dir = os.path.join(os.path.dirname(worker_script), 'tmp_extract_fragments_blockwise')
    os.makedirs(output_dir, exist_ok=True)

    log_out = os.path.join(output_dir, 'agglomerate_blockwise_%d.out' %worker_id)
    log_err = os.path.join(output_dir, 'agglomerate_blockwise_%d.err' %worker_id)

    config = {
            'pred_path': pred_path,
            'pred_dataset': pred_dataset,
            'fragments_path': fragments_path,
            'fragments_dataset': fragments_dataset,
            'db_host': db_host,
            'db_name': db_name,
            'edges_collection': edges_collection,
            'merge_function': merge_function,
            'threshold': threshold,
            'lsd_sigma': lsd_sigma,
            'mode': mode
            }

    config_str = ''.join(['%s'%(v,) for v in config.values()])
    config_hash = abs(int(hashlib.md5(config_str.encode()).hexdigest(), 16))

    config_file = os.path.join(output_dir, '%d.config'%config_hash)

    with open(config_file, 'w') as f:
        json.dump(config, f)

    logging.info('Running block with config %s...'%config_file)

    worker_command = os.path.join('.', worker_script)

    base_command = [
        f'python {worker_command} {config_file} > {log_out}'
    ]

    logging.info(f'Base command: {base_command}')

    daisy_call(base_command, log_out=log_out, log_err=log_err)


if __name__ == '__main__':

    config_file = sys.argv[1]

    with open(config_file, 'r') as f:
        config = json.load(f)

    start = time.time()

    agglomerate_blockwise(**config)

    end = time.time()

    seconds = end-start

    merge_function = config['merge_function']
    logging.info(f'Finished agglomerating with merge function: {merge_function}')
    logging.info(f'Total time to agglomerate: {seconds}')
