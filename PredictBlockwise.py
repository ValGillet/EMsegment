import daisy
import hashlib
import json
import logging
import numpy as np
import os
import pymongo
import time

from datetime import date
from funlib.persistence import open_ds, prepare_ds
from funlib.geometry import Roi, Coordinate

from emsegment.utils.block_wise_process import check_block, daisy_call

logging.basicConfig(level=logging.INFO)
logging.getLogger('pymongo').setLevel(logging.WARNING) # Hide pymongo output when debugging


def get_mask_roi(mask, source, raw_path, db_host=None):

    if isinstance(mask, bool):
        client = pymongo.MongoClient(db_host)
        db_mask = 'mask_info_' + raw_path.split('/')[-1].split('.')[0]

        assert db_mask in client.list_database_names()

        db = client[db_mask]
        blocks_data = db['block_data']

        top_left_nm = [d['top_left_nm'] for d in blocks_data.find({'block_masked_in':1})]
        bot_right_nm = [d['bot_right_nm'] for d in blocks_data.find({'block_masked_in':1})]

        begin = np.min(top_left_nm, 0)
        end = np.max(bot_right_nm, 0)

        size = end-begin
        roi = Roi(begin, size)

    elif isinstance(mask, list):
        begin_vx, shape_vx = mask
        begin = Coordinate(begin_vx) * source.voxel_size
        size = Coordinate(shape_vx) * source.voxel_size
        roi = Roi(begin, size)

    return roi


def predict_blockwise(
            model_config,
            raw_path,
            affs_path,
            db_name,
            models_per_gpu=1,
            num_cache_workers=4,
            mask_path=None,
            db_host=None,
            raw_dataset='raw',
            affs_dataset='pred_affs',
            roi_start=None,
            roi_size=None,
            GPU_pool=None):
    
    model_path      = model_config['model_path']
    num_fmaps       = model_config['num_fmaps']
    output_shape    = model_config['output_shape']
    padding         = model_config['padding']

    logging.info(f'Starting predictions for file:\n    {raw_path}\n')
    logging.info(f'Using GPUs: {GPU_pool}')
    logging.info(f'Loading {models_per_gpu} models per GPU')     
    GPU_pool = GPU_pool*models_per_gpu

    # Prepare paths
    model_path = os.path.abspath(model_path)
    raw_path   = os.path.abspath(raw_path)
    affs_path  = os.path.abspath(affs_path)

    assert os.path.exists(model_path)
    logging.info(f'Model at:\n    {model_path}\n')
    
    # Prepare raw data 
    source = open_ds(os.path.join(raw_path, raw_dataset))
    # source = open_ds(raw_path, raw_dataset)
    total_roi = source.roi
    
    if roi_start is not None and roi_size is not None:
        roi = Roi(roi_start, roi_size)
        total_roi = total_roi.intersect(roi)

    if mask_path is not None:
        # Either use mask info from the db, or crop to a given bbox ([begin_zyx, shape_zyx])
        masked_roi = get_mask_roi(mask_path, source, raw_path, db_host)
        
        logging.info(f'Cropping ROI of source: {source.roi}')
        logging.info(f'To ROI of mask: {masked_roi}')

        total_roi = total_roi.intersect(masked_roi)
    
    # Prepare variables
    voxel_size = source.voxel_size

    input_shape = Coordinate(output_shape) + Coordinate(padding)
    input_size = Coordinate(input_shape) * voxel_size  
    output_size = Coordinate(output_shape) * voxel_size

    context = (input_size - output_size) / 2
    
    read_roi = Roi((0,0,0), input_size) - context
    write_roi = Roi((0,0,0), output_size)

    # Get total ROIs (shrink total_roi)
    output_roi = total_roi.grow(-context, -context)
    output_vx_shape = output_roi.get_shape() / voxel_size

    # Prepare output
    affs = prepare_ds(
                      store=os.path.join(affs_path, affs_dataset),
                      shape=(3, *output_vx_shape),
                      offset=total_roi.begin,
                      voxel_size=voxel_size,
                      mode='a',
                      dtype=np.float32)
    
    logging.info(f'Source roi: {total_roi}')
    logging.info(f'Output roi: {affs.roi}')
    logging.info(f'Source voxel size: {source.voxel_size}')
    
    # MongoDB stuff
    client = pymongo.MongoClient(db_host)
    db = client[db_name]
    blocks_collection_name = 'blocks_predicted'
    if blocks_collection_name not in db.list_collection_names():
        blocks_predicted = db[blocks_collection_name]
        blocks_predicted.create_index(
            [('block_id', pymongo.ASCENDING)],
            name='block_id')

    # Process block-wise
    logging.info('Starting block-wise processing...')
    tasks = daisy.Task(
                task_id=f'Predict-{db_name}',
                total_roi=output_roi,
                read_roi=read_roi,
                write_roi=write_roi,
                process_function=lambda: start_predict_worker(
                                                        model_path,
                                                        num_fmaps,
                                                        raw_path,
                                                        raw_dataset,
                                                        affs_path,
                                                        affs_dataset,
                                                        input_size,
                                                        output_size,
                                                        db_host,
                                                        db_name,
                                                        num_cache_workers,
                                                        GPU_pool),
                check_function=lambda b: check_block(
                                                b, 
                                                db_host, db_name,
                                                blocks_collection_name),
                num_workers=len(GPU_pool),
                read_write_conflict=False,
                fit='overhang'
                       )
    
    done = daisy.run_blockwise([tasks])
        
    if done:
        doc = {
            'task': 'prediction',
            'date': date.today().strftime('%d%m%Y'),
            'voxel_size': list(affs.voxel_size),
            'size_roi_nm': list(total_roi.get_shape()),
            'start_roi_nm': list(total_roi.begin),
            'model_path': model_path,
            'num_fmaps': num_fmaps,
            'raw_path': raw_path,
            'raw_dataset': raw_dataset,
            'affs_path': affs_path,
            'affs_dataset': affs_dataset,
            'input_size': input_size,
            'output_size': output_size,
            'num_cache_workers': num_cache_workers,
            'GPU_pool': GPU_pool,
            }
        db['info_segmentation'].insert_one(doc)
        
    return done


def start_predict_worker(
        model_path,
        num_fmaps,
        raw_path,
        raw_dataset,
        affs_path,
        affs_dataset,
        input_size,
        output_size,
        db_host,
        db_name,
        num_cache_workers,
        GPU_pool):

    daisy_context = daisy.Context.from_env()
    worker_id = int(context.get('worker_id'))
    GPU_ID = GPU_pool[worker_id]

    worker_script = '/mnt/hdd1/SRC/EMpipelines/EMsegment/emsegment/workers/PredictWorker.py'
    
    output_dir = os.path.join(os.path.dirname(worker_script), 'tmp_predict_blockwise')
    os.makedirs(output_dir, exist_ok=True)
    
    log_out = os.path.join(output_dir, 'predict_blockwise_%d.out'%worker_id)
    log_err = os.path.join(output_dir, 'predict_blockwise_%d.err'%worker_id)
    
    config = {
        'model_path': model_path,
        'num_fmaps': num_fmaps,
        'raw_path': raw_path,
        'raw_dataset': raw_dataset,
        'output_path': affs_path,
        'output_dataset': affs_dataset,
        'input_size': input_size,
        'output_size': output_size,
        'db_host': db_host,
        'db_name': db_name,
        'num_cache_workers': num_cache_workers
    }

    config_str = ''.join(['%s'%(v,) for v in config.values()])
    config_hash = abs(int(hashlib.md5(config_str.encode()).hexdigest(), 16))

    config_file = os.path.join(output_dir, '%d.config'%config_hash)
    
    with open(config_file, 'w') as f:
        json.dump(config, f)
    
    logging.info('Running block with config %s...'%config_file)

    worker_command = [
        'CUDA_VISIBLE_DEVICES=%s python -u %s %s'%(
            GPU_ID,
            worker_script,
            config_file
        )]

    logging.debug(f'Worker command: {command}')
    # call command
    daisy_call(worker_command, log_out=log_out, log_err=log_err)

    logging.info('Predict worker finished')    


if __name__ == "__main__":

    config_file = sys.argv[1]

    with open(config_file, 'r') as f:
        config = json.load(f)

    start = time.time()

    predict_blockwise(**config)

    end = time.time()

    seconds = end - start
    logging.info(f'Total time to predict: {seconds}')
