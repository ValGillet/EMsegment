import daisy
import hashlib
import json
import logging
import numpy as np
import os

import pymongo
import sys
import time

from datetime import date
from funlib.persistence import open_ds, prepare_ds
from funlib.persistence.arrays.metadata import MetaDataFormat
from funlib.geometry import Roi, Coordinate

from emsegment.utils.block_wise_process import check_block, daisy_call

logging.basicConfig(level=logging.INFO)
logging.getLogger('pymongo').setLevel(logging.WARNING) # Hide pymongo output when debugging


def get_mask_roi(raw_path, db_name=None, db_host=None, ignore_missing=True):
    '''
    Retrieve the bounding box of all regions masked-in. 
    This is used to constrain prediction to only regions where a mask exists.

    Parameters
    ----------
    raw_path : str
        Path to raw data zarr store. Used to construct default database name if db_name=None.
    db_name : str or None, optional
        MongoDB database name containing mask info. If None, auto-generated from raw_path
        as 'mask_info_<zarr_name>'. Default: None
    db_host : str or None, optional
        MongoDB connection URI. If None, uses localhost. Default: None
    ignore_missing : bool, optional
        If True, returns None when database doesn't exist. If False, raises ValueError.
        Default: True

    Returns
    -------
    funlib.geometry.Roi or None
        Region of interest covering all masked-in blocks in physical coordinates (nm).
        Returns None if database doesn't exist and ignore_missing=True.

    Raises
    ------
    ValueError
        If database doesn't exist and ignore_missing=False.

    Notes
    -----
    - Expects MongoDB collection named 'block_data' with documents containing:
      - 'block_masked_in': 1 for included blocks
      - 'top_left_nm': [z, y, x] starting coordinates in nanometers
      - 'bot_right_nm': [z, y, x] ending coordinates in nanometers
    '''

    client = pymongo.MongoClient(db_host)

    if db_name is None:
        db_name = 'mask_info_' + raw_path.split('/')[-1].rstrip('.zarr')
    else:
        db_name = db_name

    if not db_name in client.list_database_names():
        if ignore_missing:
            return None
        else:
            raise ValueError(f'Mask database missing: {db_name}')

    db = client[db_name]
    blocks_data = db['block_data']

    top_left_nm = [d['top_left_nm'] for d in blocks_data.find({'block_masked_in':1})]
    bot_right_nm = [d['bot_right_nm'] for d in blocks_data.find({'block_masked_in':1})]

    begin = np.min(top_left_nm, 0)
    end = np.max(bot_right_nm, 0)

    size = end-begin
    return Roi(begin, size)
    

def predict_blockwise(
            model_config,
            raw_path,
            output_path,
            db_name,
            models_per_gpu=1,
            num_cache_workers=4,
            mask_path=None,
            db_host=None,
            raw_dataset='raw',
            affs_dataset='pred_affs',
            lsds_dataset='pred_lsds',
            write_affs=True,
            write_lsds=False,
            roi_start=None,
            roi_size=None,
            GPU_pool=None):
    '''
    Run affinity/LSDs prediction.
    Based on https://github.com/funkelab/lsd/blob/master/lsd/tutorial/scripts/01_predict_blockwise.py

    Parameters
    ----------
    model_config : dict
        Dictionary containing model parameters with keys:
        - 'model_path': Path to trained PyTorch model checkpoint
        - 'num_fmaps': Number of feature maps in model architecture
        - 'output_shape': Model output shape in voxels [z, y, x]
        - 'padding': Context padding required by model [z, y, x]
    raw_path : str
        Path to input zarr container with raw data.
    output_path : str
        Path to output zarr container where predictions will be written.
    db_name : str
        MongoDB database name for tracking block completion status.
    models_per_gpu : int, optional
        Number of model instances to run per GPU. Multiplies effective GPU pool size.
        Default: 1
    num_cache_workers : int, optional
        Number of gunpowder cache workers for data loading per model instance.
        Default: 4
    mask_path : str or None, optional
        Path to zarr containing binary mask. If provided, constrains prediction to masked region.
        Default: None
    db_host : str or None, optional
        MongoDB connection URI. If None, uses localhost. Default: None
    raw_dataset : str, optional
        Name of dataset in raw_path containing raw data. Default: 'raw'
    affs_dataset : str, optional
        Name of output dataset for affinity predictions. Default: 'pred_affs'
    lsds_dataset : str, optional
        Name of output dataset for LSD predictions. Default: 'pred_lsds'
    write_affs : bool, optional
        Whether to write affinity predictions. Default: True
    write_lsds : bool, optional
        Whether to write LSD predictions. Default: False
    roi_start : array-like or None, optional
        Starting coordinates [z, y, x] in nanometers for region to process.
        If None, uses full volume. Default: None
    roi_size : array-like or None, optional
        Size [z, y, x] in nanometers for region to process.
        If None, uses full volume. Default: None
    GPU_pool : list of int or None, optional
        List of CUDA device IDs to use (e.g., [0, 1, 2]).
        Will be multiplied by models_per_gpu to create worker pool. Default: None

    Returns
    -------
    bool
        True if all blocks completed successfully, False otherwise.

    Notes
    -----
    - Block size and context are automatically computed from model's output_shape and padding
    - Output datasets are created with shape (channels, z, y, x) as float32
    - Chunk shape is aligned to model output_shape to avoid artifacts
    - MongoDB collection 'blocks_predicted' tracks completion status
    - Metadata is written to 'info_segmentation' collection upon successful completion
    - Workers write logs to: workers/tmp_predict_blockwise/predict_blockwise_<id>.{out,err}
    - Requires .zgroup file in zarr for gunpowder compatibility (created if missing)

    Examples
    --------
    Run prediction with 2 GPUs, 2 models per GPU (4 workers total):

    >>> model_config = {
    ...     'model_path': '/models/em_model.pt',
    ...     'num_fmaps': 12,
    ...     'output_shape': [40, 200, 200],
    ...     'padding': [20, 100, 100]
    ... }
    >>> predict_blockwise(
    ...     model_config=model_config,
    ...     raw_path='/data/raw.zarr',
    ...     output_path='/data/predictions.zarr',
    ...     db_name='my_predictions',
    ...     models_per_gpu=2,
    ...     GPU_pool=[0, 1],
    ...     write_affs=True,
    ...     write_lsds=True
    ... )

    See Also
    --------
    workers.PredictWorker.predict : Worker function that performs actual inference
    start_predict_worker : Spawns worker subprocesses with GPU assignment
    '''
    
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
    output_path  = os.path.abspath(output_path)

    assert os.path.exists(model_path)
    logging.info(f'Model at:\n    {model_path}\n')
    
    # Prepare raw data 
    source = open_ds(os.path.join(raw_path, raw_dataset), mode='r')
    if source.voxel_size == (1,1,1):
        # If default resolution is set, it could be that a different keyword is being used
        source = open_ds(os.path.join(raw_path, raw_dataset), mode='r', 
                         metadata_format=MetaDataFormat(voxel_size_attr='resolution'))
        if source.voxel_size == (1,1,1):
            logging.warning('Voxel size appears to be missing from input dataset attributes. Using (1,1,1).')
    
    if not os.path.exists(os.path.join(raw_path, '.zgroup')):
        # gunpowder node ZarrSource needs the .zgroup file
        with open(os.path.join(raw_path, '.zgroup'), 'w') as f:
            json.dump({'zarr_format': 2}, f)

    total_roi = source.roi
    
    # Constrain start and/or size to what was provided
    if roi_start is None:
        roi_start = total_roi.begin
    if roi_size is None:
        roi_size = total_roi.end - roi_start
    total_roi = total_roi.intersect(Roi(roi_start, roi_size))

    if mask_path is not None:
        # Either use mask info from the db, or crop to a given bbox ([begin_zyx, shape_zyx])
        masked_roi = get_mask_roi(raw_path, db_host=db_host, ignore_missing=True)
        
        if masked_roi is None:
            logging.info('Mask database missing. Will use the full ROI.')
        else:
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
    input_roi = total_roi.grow(context, context)
    output_roi = total_roi
    output_roi_shape = output_roi.get_shape() / voxel_size

    # Prepare output
    if write_affs:
        store_path = os.path.join(output_path, affs_dataset)
        affs = prepare_ds(
                        store=os.path.join(output_path, affs_dataset),
                        shape=(3, *output_roi_shape),
                        offset=total_roi.begin,
                        voxel_size=voxel_size,
                        axis_names=['c^','z','y','x'],
                        units=['nm','nm','nm'],
                        mode='a',
                        dtype=np.float32,
                        chunk_shape=Coordinate(3, *output_shape), # Chunk shape needs to be write-aligned or we end up with black patches
                        custom_metadata={'resolution': list(voxel_size)} # For compatibility 
                        )
    if write_lsds:
        store_path = os.path.join(output_path, lsds_dataset)
        lsds = prepare_ds(
                        store=os.path.join(output_path, lsds_dataset),
                        shape=(10, *output_roi_shape),
                        offset=total_roi.begin,
                        voxel_size=voxel_size,
                        axis_names=['c^','z','y','x'],
                        units=['nm','nm','nm'],
                        mode='a',
                        dtype=np.float32,
                        chunk_shape=Coordinate(11, *output_shape), # Chunk shape needs to be write-aligned or we end up with black patches
                        custom_metadata={'resolution': list(voxel_size)} # For compatibility 
                        )
    
    logging.info(f'Source roi: {total_roi}')
    logging.info(f'Input roi: {input_roi}')
    logging.info(f'Output roi: {output_roi}')
    logging.info(f'Source voxel size: {source.voxel_size}')
    logging.info(f'Read ROI: {read_roi}')
    logging.info(f'Write ROI: {write_roi}')
    if write_affs:
        logging.info('Writing affinities')
    if write_lsds:
        logging.info('Writing LSDs')
    
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
                total_roi=input_roi,
                read_roi=read_roi,
                write_roi=write_roi,
                process_function=lambda: start_predict_worker(
                                                        model_path,
                                                        num_fmaps,
                                                        raw_path,
                                                        raw_dataset,
                                                        output_path,
                                                        affs_dataset,
                                                        lsds_dataset,
                                                        write_affs,
                                                        write_lsds,
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
            'voxel_size': list(source.voxel_size),
            'size_roi_nm': list(total_roi.get_shape()),
            'start_roi_nm': list(total_roi.begin),
            'model_path': model_path,
            'num_fmaps': num_fmaps,
            'raw_path': raw_path,
            'raw_dataset': raw_dataset,
            'output_path': output_path,
            'affs_dataset': affs_dataset,
            'lsds_dataset': lsds_dataset,
            'write_affs': write_affs,
            'write_lsds': write_lsds,
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
        output_path,
        affs_dataset,
        lsds_dataset,
        write_affs,
        write_lsds,
        input_size,
        output_size,
        db_host,
        db_name,
        num_cache_workers,
        GPU_pool):
    '''
    Spawn a PredictWorker subprocess with GPU assignment for block-wise inference.

    Called by daisy for each worker process. Creates a configuration file, assigns a GPU
    from the pool based on worker ID, and launches PredictWorker.py as a subprocess with
    CUDA_VISIBLE_DEVICES set appropriately.

    Parameters
    ----------
    model_path : str
        Path to trained PyTorch model checkpoint file.
    num_fmaps : int
        Number of feature maps in model architecture.
    raw_path : str
        Path to input zarr container with raw EM data.
    raw_dataset : str
        Name of dataset in raw_path containing EM data.
    output_path : str
        Path to output zarr container for predictions.
    affs_dataset : str
        Name of output dataset for affinity predictions.
    lsds_dataset : str
        Name of output dataset for LSD predictions.
    write_affs : bool
        Whether to generate and write affinity predictions.
    write_lsds : bool
        Whether to generate and write LSD predictions.
    input_size : funlib.geometry.Coordinate
        Input size for model including context, in physical units (nm).
    output_size : funlib.geometry.Coordinate
        Output size for model (write ROI), in physical units (nm).
    db_host : str
        MongoDB connection URI.
    db_name : str
        MongoDB database name for tracking progress.
    num_cache_workers : int
        Number of gunpowder cache workers for data loading.
    GPU_pool : list of int
        List of CUDA device IDs. Worker selects GPU using: GPU_pool[worker_id].

    Notes
    -----
    - Worker ID is obtained from daisy.Context.from_env()
    - Worker logs written to: workers/tmp_predict_blockwise/predict_blockwise_<id>.{out,err}
    - GPU assignment: worker_id % len(GPU_pool) maps to specific GPU
    - Configuration file persists for debugging (not auto-deleted)

    See Also
    --------
    workers.PredictWorker.predict : Worker subprocess that performs inference
    '''

    daisy_context = daisy.Context.from_env()
    worker_id = int(daisy_context.get('worker_id'))
    GPU_ID = GPU_pool[worker_id]

    worker_script = os.path.join(os.path.dirname(__file__), 'workers', 'PredictWorker.py')

    output_dir = os.path.join(os.path.dirname(worker_script), 'tmp_predict_blockwise')
    os.makedirs(output_dir, exist_ok=True)
    
    log_out = os.path.join(output_dir, 'predict_blockwise_%d.out'%worker_id)
    log_err = os.path.join(output_dir, 'predict_blockwise_%d.err'%worker_id)
    
    config = {
        'model_path': model_path,
        'num_fmaps': num_fmaps,
        'raw_path': raw_path,
        'raw_dataset': raw_dataset,
        'output_path': output_path,
        'input_size': input_size,
        'output_size': output_size,
        'db_host': db_host,
        'db_name': db_name,
        'num_cache_workers': num_cache_workers,
        'affs_dataset': affs_dataset,
        'lsds_dataset': lsds_dataset,
        'write_affs': write_affs,
        'write_lsds': write_lsds
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

    logging.debug(f'Worker command: {worker_command}')
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
