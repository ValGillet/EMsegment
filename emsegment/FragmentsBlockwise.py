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
    Extract fragments (supervoxels) from predictions using watershed segmentation.
    Based on https://github.com/funkelab/lsd/blob/master/lsd/tutorial/scripts/02_extract_fragments_blockwise.py

    Parameters
    ----------
    pred_path : str
        Path to zarr container with prediction data (affinities or LSDs).
    chunk_voxel_size : list of int
        Block size in voxels [z, y, x] for processing.
    context_px : list of int
        Context (overlap) size in voxels [z, y, x] for block boundaries.
    db_name : str
        MongoDB database name for tracking block completion.
    num_workers : int
        Number of CPU worker processes for parallel watershed.
    db_host : str or None, optional
        MongoDB connection URI. If None, uses localhost. Default: None
    pred_dataset : str, optional
        Name of dataset in pred_path containing predictions. Auto-detects mode:
        3 channels = affinities, 10 channels = LSDs. Default: 'pred_affs'
    fragments_path : str or None, optional
        Path to output zarr for fragments. If None, uses pred_path. Default: None
    fragments_dataset : str, optional
        Name of output dataset for fragments. Default: 'frags'
    mask_path : str or None, optional
        Path to zarr containing binary mask to constrain watershedding. Default: None
    mask_dataset : str, optional
        Name of mask dataset in mask_path. Default: 'mask'
    fragments_in_xy : bool, optional
        If True, perform 2D watershed on XY slices. If False, 3D watershed. Default: True
    epsilon_agglomerate : float, optional
        Initial agglomeration threshold applied during watershed to merge fragments
        below this threshold. Default: 0 (no merging)
    filter_fragments : float, optional
        Size threshold for filtering small fragments (in voxels). Default: 0 (no filtering)
    min_seed_distance : int, optional
        Minimum distance between watershed seeds in voxels. Lower values create denser
        supervoxels. Default: 5
    lsd_sigma : float, optional
        Sigma parameter for Gaussian smoothing when using LSD mode, in nanometers.
        Only used if pred_dataset contains LSDs. Default: 60.0
    replace_sections : list or None, optional
        List of Z-sections to replace/reprocess. Default: None (process all)

    Returns
    -------
    bool
        True if all blocks completed successfully, False otherwise.

    Notes
    -----
    - Automatically detects mode from prediction shape: 3 channels=affs, 10 channels=LSDs
    - Fragments are stored as uint64 labels
    - MongoDB collection 'blocks_fragments' tracks completion status
    - Metadata written to 'info_segmentation' collection upon successful completion
    - Workers write logs to: workers/tmp_extract_fragments_blockwise/extract_fragments_blockwise_<id>.{out,err}

    Examples
    --------
    Extract fragments using affinity predictions with 2D watershed:

    >>> extract_fragments_blockwise(
    ...     pred_path='/data/predictions.zarr',
    ...     chunk_voxel_size=[100, 500, 500],
    ...     context_px=[10, 50, 50],
    ...     db_name='my_fragments',
    ...     num_workers=8,
    ...     pred_dataset='pred_affs',
    ...     fragments_in_xy=True,
    ...     min_seed_distance=5
    ... )

    Extract denser supervoxels with smaller seed distance and filtering:

    >>> extract_fragments_blockwise(
    ...     pred_path='/data/predictions.zarr',
    ...     chunk_voxel_size=[100, 500, 500],
    ...     context_px=[10, 50, 50],
    ...     db_name='my_fragments',
    ...     num_workers=8,
    ...     min_seed_distance=3,
    ...     filter_fragments=0.5,
    ...     epsilon_agglomerate=0.05
    ... )

    See Also
    --------
    workers.FragmentsWorker.extract_fragments_worker : Worker that performs watershed
    utils.lsds.watershed_in_block_affs : Affinity-based watershed implementation
    utils.lsds.watershed_in_block_lsds : LSD-based watershed implementation
    start_frag_worker : Spawns worker subprocesses
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
    '''
    Spawn a FragmentsWorker subprocess for block-wise watershed segmentation.
    Called by daisy for each worker process. 

    Parameters
    ----------
    pred_path : str
        Full path to prediction dataset (e.g., '/path/to/data.zarr/pred_affs').
    fragments_path : str
        Full path to output fragments dataset (e.g., '/path/to/data.zarr/frags').
    db_host : str
        MongoDB connection URI.
    db_name : str
        MongoDB database name for tracking progress.
    context : funlib.geometry.Coordinate
        Context size in physical units (nm) for block overlap.
    fragments_in_xy : bool
        If True, perform 2D watershed on XY slices. If False, 3D watershed.
    num_voxels_in_block : int
        Total number of voxels in write ROI (used for fragment ID offset).
    epsilon_agglomerate : float
        Initial agglomeration threshold for merging fragments during watershed.
    mask_path : str or None
        Full path to mask dataset if using masking, None otherwise.
    filter_fragments : float
        Size threshold for filtering small fragments (in voxels).
    min_seed_distance : int
        Minimum distance between watershed seeds in voxels.
    replace_sections : list or None
        List of Z-sections to replace/reprocess.
    lsd_sigma : float
        Sigma for Gaussian smoothing when using LSD mode (in nm).
    mode : str
        Watershed mode, either 'affs' or 'lsds'.

    Notes
    -----
    - Worker ID is obtained from daisy.Context.from_env()
    - Worker logs written to: workers/tmp_extract_fragments_blockwise/extract_fragments_blockwise_<id>.{out,err}
    - Configuration file persists for debugging

    See Also
    --------
    workers.FragmentsWorker.extract_fragments_worker : Worker subprocess implementation
    '''
   
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
