import daisy
import hashlib
import json
import logging
import lsd
import numpy as np
import os
import pymongo
import sys
import time

from datetime import date
from funlib.persistence import open_ds, prepare_ds
from funlib.geometry import Coordinate

from emsegment.utils.block_wise_process import check_block, daisy_call

logging.basicConfig(level=logging.INFO)
logging.getLogger('pymongo').setLevel(logging.WARNING) # Hide pymongo output when debugging


def extract_fragments_blockwise(
                      pred_path,
                      chunk_voxel_size,
                      context_px,
                      db_name,
                      num_workers,
                      db_host=None,
                      pred_dataset='pred_affs',
                      fragments_path=None,
                      fragments_dataset='frags',
                      mask_path=None,
                      mask_dataset='mask',
                      fragments_in_xy=True,
                      epsilon_agglomerate=0,
                      filter_fragments=0,
                      min_seed_distance=5,
                      lsd_sigma=60.0,
                      replace_sections=None):
    
    '''
    min_seed_distance: Distance between seeds for watershedding. Influences the density of supervoxels
    '''
    
    fragments_path = pred_path if fragments_path is None else fragments_path
    mask_path = os.path.join(mask_path, mask_dataset) if mask_path is not None else mask_path

    logging.info(f'Reading predictions from {pred_path}')
    logging.info(f'Using dataset: {pred_dataset}')
    pred = open_ds(os.path.join(pred_path, pred_dataset))

    if pred.shape[0] == 3:
        mode = 'affs'
    elif pred.shape[0] == 10:
        mode = 'lsds'
    else:
        raise ValueError(f'Unexpected shape for prediction dataset: {pred.shape}')

    # Prepare variables
    voxel_size = pred.voxel_size 
    chunk_size = Coordinate(chunk_voxel_size) * voxel_size
    context = Coordinate(context_px) * voxel_size

    read_roi = daisy.Roi((0,0,0), chunk_size).grow(context, context)
    write_roi = daisy.Roi((0,0,0), chunk_size)

    # Get number of voxels in block
    num_voxels_in_block = (write_roi/pred.voxel_size).get_size()    

    # Prepare fragment dataset
    store_path = os.path.join(fragments_path, fragments_dataset)
    fragments = prepare_ds(
                           store=store_path,
                           shape=pred.roi.get_shape() / voxel_size,
                           offset=pred.roi.begin,
                           voxel_size=voxel_size,
                           axis_names=['z','y','x'],
                           units=['nm','nm','nm'],
                           mode='a',
                           chunk_shape=Coordinate(chunk_voxel_size),
                           dtype=np.uint64,
                           custom_metadata={'resolution': list(voxel_size)} # For compatibility 
                           )

    # Pad roi
    total_roi = pred.roi.grow(context, context)

    # Prepare MongoDB to log blocks
    client = pymongo.MongoClient(db_host)
    db = client[db_name]
    blocks_collection_name = 'blocks_fragments'
    if blocks_collection_name not in db.list_collection_names():
        blocks_extracted = db[blocks_collection_name]
        blocks_extracted.create_index(
                                      [('block_id', pymongo.ASCENDING)],
                                      name = 'block_id')
    
    logging.info(f'Chunk voxel size: {chunk_voxel_size}')
    logging.info(f'Chunk nm size: {chunk_size}')

    # Blockwise watershed
    tasks = daisy.Task( 
                task_id=f'Fragments-{db_name}',
                total_roi=total_roi,
                read_roi=read_roi,
                write_roi=write_roi,
                process_function=lambda: start_frag_worker(
                                                     os.path.join(pred_path, pred_dataset),
                                                     os.path.join(fragments_path, fragments_dataset),
                                                     db_host,
                                                     db_name,
                                                     context,
                                                     fragments_in_xy,
                                                     num_voxels_in_block,
                                                     epsilon_agglomerate,
                                                     mask_path,
                                                     filter_fragments,
                                                     min_seed_distance,
                                                     replace_sections,
                                                     lsd_sigma,
                                                     mode
                                                     ),
                check_function=lambda b: check_block(
                                        b, 
                                        db_host, db_name,
                                        blocks_collection_name),
                num_workers=num_workers,
                read_write_conflict=False,
                fit='shrink')
    
    done = daisy.run_blockwise([tasks])

    if done:
        doc = {
            'task': 'fragments',
            'date': date.today().strftime('%d%m%Y'),
            'voxel_size': list(pred.voxel_size),
            'size_roi_nm': list(total_roi.get_shape()),
            'start_roi_nm': list(total_roi.begin),
            'pred_path': pred_path,
            'pred_dataset': pred_dataset,
            'fragments_path': fragments_path,
            'fragments_dataset': fragments_dataset,
            'chunk_voxel_size': chunk_voxel_size,
            'context': context,
            'fragments_in_xy': fragments_in_xy,
            'num_voxels_in_block': num_voxels_in_block,
            'epsilon_agglomerate': epsilon_agglomerate,
            'mask_path': mask_path,
            'filter_fragments': filter_fragments,
            'min_seed_distance': min_seed_distance,
            'replace_sections': replace_sections,
            'lsd_sigma': lsd_sigma,
            'mode': mode
              }
        db['info_segmentation'].insert_one(doc)
    
    return done

def start_frag_worker(
                 pred_path,
                 fragments_path,
                 db_host,
                 db_name,
                 context,
                 fragments_in_xy,
                 num_voxels_in_block,
                 epsilon_agglomerate,
                 mask_path,
                 filter_fragments,
                 min_seed_distance,
                 replace_sections,
                 lsd_sigma,
                 mode):
   
    daisy_context = daisy.Context.from_env()
    worker_id = int(daisy_context.get('worker_id'))
    logging.info(f'Worker {worker_id} started...')

    worker_script = os.path.join(os.path.dirname(__file__), 'workers', 'FragmentsWorker.py')

    output_dir = os.path.join(os.path.dirname(worker_script), 'tmp_extract_fragments_blockwise')
    os.makedirs(output_dir, exist_ok=True)

    log_out = os.path.join(output_dir, 'extract_fragments_blockwise_%d.out' %worker_id)
    log_err = os.path.join(output_dir, 'extract_fragments_blockwise_%d.err' %worker_id)

    config = {
            'pred_path': pred_path,
            'fragments_path': fragments_path,
            'db_host': db_host,
            'db_name': db_name,
            'context': context,
            'fragments_in_xy': fragments_in_xy,
            'num_voxels_in_block': num_voxels_in_block,
            'epsilon_agglomerate': epsilon_agglomerate,
            'mask_path': mask_path,
            'filter_fragments': filter_fragments,
            'min_seed_distance': min_seed_distance,
            'replace_sections': replace_sections,
            'lsd_sigma': lsd_sigma,
            'mode': mode
        }

    config_str = ''.join(['%s'%(v,) for v in config.values()])
    config_hash = abs(int(hashlib.md5(config_str.encode()).hexdigest(), 16))

    config_file = os.path.join(output_dir, '%d.config'%config_hash)

    with open(config_file, 'w') as f:
        json.dump(config, f)

    logging.info('Running block with config %s...'%config_file)

    worker_command = os.path.abspath(worker_script)

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

    extract_fragments_blockwise(**config)

    end = time.time()

    seconds = end - start
    logging.info(f'Total time to extract fragments: {seconds}')
