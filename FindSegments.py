import daisy
import json
import logging
import multiprocessing as mp
import numpy as np
import os
import sys
import time

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
                  chunk_voxel_size,
                  num_workers=1,
                  db_host=None,
                  fragments_dataset='frags',
                  chunk_bbox=[],
                  run_type=None,
                  **kwargs):

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
                    task_id = 'test',
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
    for threshold in thresholds:

        get_connected_components(
                nodes,
                edges,
                scores,
                threshold,
                edges_collection,
                out_dir)

        logging.info(f'Created and stored lookup tables in {time.time() - start}')


def get_connected_components(
        nodes,
        edges,
        scores,
        threshold,
        edges_collection,
        out_dir,
        **kwargs):

    logging.info(f'Getting CCs for threshold {threshold}...')
    components = connected_components(nodes, edges, scores, threshold)
    
    logging.info(f'Creating fragment-segment LUT for threshold {threshold}...')
    lut = np.array([nodes, components])

    logging.info(f'Storing fragment-segment LUT for threshold {threshold}...')

    lookup = f'seg_{edges_collection}_{int(threshold*100)}'

    out_file = os.path.join(out_dir, lookup)

    np.savez_compressed(out_file, fragment_segment_lut=lut)


def read_chunk_graph(block, fragments, db_host, db_name, edges_collection, shared_list):

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