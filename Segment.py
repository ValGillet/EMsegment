import argparse
import json
import logging
import os
import sys
import time

from datetime import date
from glob import glob
from pymongo import MongoClient

from emsegment.PredictBlockwise import predict_blockwise
from emsegment.FragmentsBlockwise import extract_fragments_blockwise
from emsegment.AgglomerateBlockwise import agglomerate_blockwise


logging.basicConfig(level=logging.DEBUG)
logging.getLogger('pymongo').setLevel(logging.WARNING) # Hide pymongo output when debugging

def segment_dataset(
                project_dir,
                project_prefix,
                model_config,
                input_path,
                GPU_pool,
                num_workers,
                seg_config='seg_config.json',
                todo=['predict', 'fragment', 'agglomerate'],
                chunk_voxel_size=[100,500,500],
                volume_suffix='',
                roi_start=None,
                roi_size=None,
                db_host=None,
                affs_path=None,
                fragments_path=None,
                use_mask=False,
                mask_path=None,
                raw_dataset='raw',
                mask_dataset='mask',
                affs_dataset='pred_affs',
                fragments_dataset='frags',
                start_over=False,
                continue_previous=False,
                return_config=False
                   ):

    ### Prepare file paths ###
    # By default all outputs go to the same zarr store in different datasets
    # If no affs_path is provided, the output will have the same name as the raw dataset
    project_dir = os.path.abspath(project_dir)
    raw_path = os.path.abspath(input_path)
    store_name = os.path.basename(raw_path).rstrip('.zarr')
    store_name = project_prefix + '_' + store_name if project_prefix else store_name
    store_name = store_name + '_' + volume_suffix if volume_suffix else store_name

    if use_mask and mask_path is None:
        # If no mask_path is specified, we assume it exists as a dataset in the raw image store
        assert os.path.exists(os.path.join(raw_path, 'mask'))
        mask_path = raw_path
        logging.info('Will use mask')

    if affs_path is None:
        # Give the same name as the raw dataset and add an index to differentiate different projects with the same input name
        affs_path = os.path.join(project_dir, store_name)

    index = str(len(glob(affs_path + '*')))

    if not continue_previous or index == '0':
        logging.info('Starting new project from scratch...')
        # Start this dataset from scratch and create its directory and config file
        seg_config = os.path.abspath(seg_config)

        affs_path += '_' + index.zfill(2) + '.zarr'
        affs_path = os.path.abspath(affs_path)
        db_name = os.path.basename(affs_path).rstrip('.zarr')
        os.makedirs(affs_path, exist_ok=True)

        fragments_path = affs_path if fragments_path is None else os.path.abspath(fragments_path)

        ### Get config parameters ###
        with open(seg_config, 'r') as f:
            seg_config = json.load(f)
            
        if isinstance(model_config, str):
            with open(model_config, 'r') as f:
                model_config = json.load(f)

        seg_config['affs_path'] = affs_path
        seg_config['affs_dataset'] = affs_dataset
        seg_config['fragments_path'] = fragments_path
        seg_config['fragments_dataset'] = fragments_dataset

        # Copy config to the project store
        seg_config.update({'db_name': db_name,
                           'raw_path': raw_path,
                           'affs_path': affs_path,
                           'fragments_path': fragments_path,
                           'chunk_voxel_size': chunk_voxel_size,
                           'roi_start': roi_start,
                           'roi_size': roi_size,
                           'model_config': model_config,
                           'start_date': date.today().strftime('%d%m%Y')})
        with open(os.path.join(affs_path, 'seg_config.json'), 'w') as f:
            json.dump(seg_config, f, indent='')

    else:
        logging.info('Continuing where we left off...')
        # Pick up where we left off with the latest directory and config file  
        index = str(max(0, int(index)-1))
        affs_path += '_' + index.zfill(2) + '.zarr'

        with open(os.path.join(affs_path, 'seg_config.json'), 'r') as f:
            seg_config = json.load(f)

        db_name = seg_config['db_name']
        fragments_path = seg_config['fragments_path']
        chunk_voxel_size = seg_config['chunk_voxel_size']
        roi_start = seg_config['roi_start']
        roi_size = seg_config['roi_size']
        model_config = seg_config['model_config']

    ### Prepare relevant variables ###
    models_per_gpu     = seg_config['affs_config']['models_per_gpu']
    num_cache_workers  = seg_config['affs_config']['num_cache_workers']

    # TODO: add possiblity to choose int or float for prediction

    context_px           = seg_config['frag_config']['context_px']
    fragments_in_xy      = seg_config['frag_config']['fragments_in_xy']
    epsilon_agglomerate  = seg_config['frag_config']['epsilon_agglomerate']
    filter_fragments     = seg_config['frag_config']['filter_fragments']
    min_seed_distance    = seg_config['frag_config']['min_seed_distance']

    agglomerate_threshold      = seg_config['agglo_config']['threshold']
    edges_collection_basename  = seg_config['agglo_config']['edges_collection_basename']
    merge_function             = seg_config['agglo_config']['merge_function']

    # MongoDB
    client = MongoClient(db_host)
    db = client[db_name]

    logging.info(f'Progress and segmentation will be stored in :\n    Path: {affs_path}\n    DB: {db_name}')

    ### Prediction ###
    if 'predict' not in todo:
        logging.info('Skipping predictions.')
    elif db.info_segmentation.find_one({'task': 'prediction'}) is not None and not start_over:
        logging.info('Predictions were already computed!')
    else:
        if start_over:
            try:
                db.blocks_predicted.drop()
                logging.info('Prediction progress was wiped.')
            except:
                logging.debug('STARTING OVER BUT DB CHECK BLOCK PREDICTIONS ALREADY EMPTY')
        print('\n----- PREDICTION -----')
        if not predict_blockwise(
                            model_config=model_config,
                            raw_path=raw_path,
                            affs_path=affs_path,
                            db_name=db_name,
                            models_per_gpu=models_per_gpu,
                            num_cache_workers=num_cache_workers, 
                            mask_path=mask_path,
                            db_host=db_host,
                            raw_dataset=raw_dataset,
                            affs_dataset=affs_dataset,
                            roi_start=roi_start,
                            roi_size=roi_size,
                            GPU_pool=GPU_pool):
            sys.exit('Interrupted or something went wrong')


    ### Fragments ###
    if 'fragment' not in todo:
        logging.info('Skipping fragments.')
    elif db.info_segmentation.find_one({'task': 'fragments'}) is not None and not start_over:
        logging.info('Fragments were already computed!')
    else:
        if start_over:
            try:
                db.blocks_predicted.drop()
                logging.info('Fragments progress was wiped.')
            except:
                logging.debug('STARTING OVER BUT DB CHECK BLOCK FRAGMENTS ALREADY EMPTY')
        print('\n----- FRAGMENTS -----')
        if not extract_fragments_blockwise(
                            affs_path=affs_path,
                            chunk_voxel_size=chunk_voxel_size,
                            context_px=context_px,
                            db_name=db_name,
                            num_workers=num_workers,
                            db_host=db_host,
                            affs_dataset=affs_dataset,
                            fragments_path=fragments_path,
                            fragments_dataset=fragments_dataset,
                            mask_file=mask_path,
                            mask_dataset=mask_dataset,
                            fragments_in_xy=fragments_in_xy,
                            epsilon_agglomerate=epsilon_agglomerate,
                            filter_fragments=filter_fragments,
                            min_seed_distance=min_seed_distance,
                            replace_sections=None):
            sys.exit('Interrupted or something went wrong')
    

    ### Agglomeration ###
    if 'agglomerate' not in todo:
        logging.info('Skipping agglomeration.')
    elif db.info_segmentation.find_one({'task': 'agglomeration'}) is not None and not start_over:
        logging.info('Agglomeration was already done!')
    else:
        if start_over:
            try:
                db.blocks_predicted.drop()
                logging.info('Prediction progress was wiped.')
            except:
                logging.debug('STARTING OVER BUT DB CHECK BLOCK AGGLOMERATION ALREADY EMPTY')
        print('\n----- AGGLOMERATION -----')
        if not agglomerate_blockwise(
                            affs_path=affs_path,
                            chunk_voxel_size=chunk_voxel_size,
                            context_px=context_px,
                            db_name=db_name,
                            merge_function=merge_function,
                            num_workers=num_workers,
                            db_host=db_host,
                            affs_dataset=affs_dataset,
                            fragments_path=fragments_path,
                            fragments_dataset=fragments_dataset,
                            edges_collection=edges_collection_basename,
                            threshold=agglomerate_threshold,
                        ):
            sys.exit('Interrupted or something went wrong')
        
    logging.info('Segmentation is complete!')
    logging.info(f'Project dir:\n    {project_dir}')
    logging.info(f'DB name: {db_name}\n')

    if return_config:
         return seg_config


if __name__ == '__main__':

    parser=argparse.ArgumentParser('')
    # Required
    parser.add_argument('-p', '--project-dir',
                        metavar='PROJECT_DIR',
                        dest='project_dir',
                        required=True,
                        type=str,
                        help='Path to the directory where to write the outputs.\
                              If not provided, the outputs will take the same name as the directory')
    parser.add_argument('-prefix', '--project-prefix',
                        metavar='PROJECT_PREFIX',
                        dest='project_prefix',
                        default='',
                        type=str,
                        help='Prefix used to name the project files.')
    parser.add_argument('-i', '--input-path',
                        metavar='INPUT_PATH',
                        dest='input_path',
                        required=True,
                        type=str,
                        help='Path to the zarr container where the image data to segment is stored.')
    parser.add_argument('-m', '--model-config',
                        metavar='MODEL_CONFIG',
                        dest='model_config',
                        required=True,
                        type=str,
                        help='Path to the json file containing the model\'s configuration.')
    parser.add_argument('-c', '--cores',
                        metavar='CORES',
                        dest='num_workers',
                        required=True,
                        type=int,
                        help='Number of threads to use for processing. 0 = all cores available')
    parser.add_argument('--GPU',
                        metavar='GPU_POOL',
                        dest='GPU_pool',
                        required=True,
                        nargs='+',
                        type=int,
                        help='CUDA device ID for the GPUs to be used for prediction.')
    
    # Not required
    parser.add_argument('--seg-config',
                        metavar='SEG_CONFIG',
                        dest='seg_config',
                        type=str,
                        default='seg_config.json',
                        help='Path to a JSON file containing parameters for the different segmentation steps.')
    parser.add_argument('--todo',
                        metavar='TODO',
                        dest='todo',
                        nargs='+',
                        type=str,
                        default=['predict', 'fragment', 'agglomerate'],
                        help='List of task to do. Up to 3 of: predict, fragment, agglomerate')
    parser.add_argument('--chunk-voxel-size',
                        metavar='CHUNK_VOXEL_SIZE',
                        dest='chunk_voxel_size',
                        nargs=3,
                        type=int,
                        default=[100,500,500],
                        help='Size of chunks (ZYX in voxels) to process in parallel.')
    parser.add_argument('--volume-suffix',
                        metavar='VOLUME_SUFFIX',
                        dest='volume_suffix',
                        type=str,
                        default='',
                        help='Prefix used to name the project files.')
    parser.add_argument('--roi-start',
                        metavar='ROI_START',
                        dest='roi_start',
                        nargs=3,
                        type=int,
                        default=None,
                        help='Starting coordinates (ZYX in nm) of the region to segment, contained in the input image stack.')
    parser.add_argument('--roi-size',
                        metavar='ROI_SIZE',
                        dest='roi_size',
                        nargs=3,
                        type=int,
                        default=None,
                        help='Size (ZYX in nm) of the region to segment, contained in the input image stack.')
    parser.add_argument('--db-host',
                        metavar='DB_HOST',
                        dest='db_host',
                        type=str,
                        default=None,
                        help='URI to the MongoDB where to store information. Default: None (localhost)')
    parser.add_argument('--affs-path',
                        metavar='AFFS_PATH',
                        dest='affs_path',
                        type=str,
                        default=None,
                        help='Path to the zarr container where to write affinity predictions.\
                              Will be created in the project dir with the same name if not provided.')
    parser.add_argument('--fragments-path',
                        metavar='FRAGMENTS_PATH',
                        dest='fragments_path',
                        type=str,
                        default=None,
                        help='Path to the zarr container where to write fragments.\
                              Will be stored in the same container as affinities if not provided.')
    parser.add_argument('--mask-path',
                        metavar='MASK_PATH',
                        dest='mask_path',
                        default=None,
                        help='Path to the zarr container where a mask exists (1 = segment).\
                              If True (boolean, not evaluating as True), we assume the mask dataset to be contained in the image container.')
    parser.add_argument('--raw-dataset',
                        metavar='RAW_DATASET',
                        dest='raw_dataset',
                        type=str,
                        default='raw',
                        help='Name of the dataset containing the raw data. Default: raw')
    parser.add_argument('--mask-dataset',
                        metavar='MASK_DATASET',
                        dest='mask_dataset',
                        type=str,
                        default='mask',
                        help='Name of the dataset containing the mask data. Default: mask')
    parser.add_argument('--affs-dataset',
                        metavar='AFFS_DATASET',
                        dest='affs_dataset',
                        type=str,
                        default='pred_affs',
                        help='Name of the dataset containing the affinity prediction data. Default: pred_affs')
    parser.add_argument('--fragments-dataset',
                        metavar='FRAGMENTS_DATASET',
                        dest='fragments_dataset',
                        type=str,
                        default='frags',
                        help='Name of the dataset containing the fragments data. Default: frags')
    parser.add_argument('--start-over',
                        action='store_true',
                        dest='start_over',
                        default=False,
                        help='If True, wipe the progress in MongoDB and start over all tasks in the todo list.')
    parser.add_argument('--continue-previous',
                        action='store_true',
                        dest='continue_previous',
                        default=False,
                        help='If True, continue the previous task found in project dir.')
    args=parser.parse_args()

    start = time.time()

    segment_dataset(**vars(args))

    end = time.time()

    seconds = (end - start)%60
    minutes = (end - start)//60
    
    logging.info(f'Total time to segment: {minutes}min{seconds}')
