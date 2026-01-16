import daisy
import json
import logging
import pymongo
import sys
import time

from daisy.block import BlockStatus
from funlib.persistence import open_ds
from lsd.post.persistence.mongodb_rag_provider import MongoDbRagProvider
from emsegment.utils.lsds import watershed_in_block_affs, watershed_in_block_lsds


logging.basicConfig(level = logging.DEBUG)
logging.getLogger('pymongo').setLevel(logging.WARNING) # Hide pymongo output when debugging

def extract_fragments_worker(input_config):

    logging.info(sys.argv)

    with open(input_config, 'r') as f:
        config = json.load(f)

    logging.info(config)
    
    # Read config (obsolete if with main script)
    pred_path            = config['pred_path']
    fragments_path       = config['fragments_path']
    db_host              = config['db_host']
    db_name              = config['db_name']
    num_voxels_in_block  = config['num_voxels_in_block']
    mask_path            = config['mask_path']
    mode                 = config['mode']

    # For affinities
    context              = config['context']
    fragments_in_xy      = config['fragments_in_xy']
    epsilon_agglomerate  = config['epsilon_agglomerate']
    filter_fragments     = config['filter_fragments']
    replace_sections     = config['replace_sections']    
    min_seed_distance    = config['min_seed_distance']

    # For lsds
    sigma = config.get('lsd_sigma', 60.0)
    
    # Open files
    logging.info(f'Reading predictions from {pred_path}')
    pred = open_ds(pred_path, mode = 'r')
   
    logging.info(f'Reading fragments from {fragments_path}')
    fragments = open_ds(fragments_path, mode='r+')

    if mask_path and isinstance(mask_path, str):
        logging.info(f'Reading mask from {mask_path}')
        mask = open_ds(mask_path)
    else:
        mask = None

    # Open RAG DB
    logging.info('Opening RAG DB...')
    rag_provider = MongoDbRagProvider(db_name,
                                      host=db_host,
                                      mode='r+')
    logging.info('RAG DB opened')

    if mode == 'affs':
        assert pred.shape[0] == 3, f'Unexpected shape for the affinity dataset: {pred.shape}'
        watershed_fun = watershed_in_block_affs
        args = {
            'affs': pred,
            'context': context,
            'rag_provider': rag_provider,
            'fragments_out': fragments,
            'num_voxels_in_block': num_voxels_in_block,
            'mask': mask,
            'fragments_in_xy': fragments_in_xy,
            'epsilon_agglomerate': epsilon_agglomerate,
            'filter_fragments': filter_fragments,
            'min_seed_distance': min_seed_distance,
            'replace_sections': replace_sections
        }
    elif mode == 'lsds':
        assert pred.shape[0] == 10, f'Unexpected shape for the LSDs dataset: {pred.shape}'
        watershed_fun = watershed_in_block_lsds
        args = {
            'lsds': pred,
            'sigma': sigma,
            'rag_provider': rag_provider,
            'fragments_out': fragments,
            'num_voxels_in_block': num_voxels_in_block,
            'mask': mask
        }

    # Open extracted blocks DB
    client = pymongo.MongoClient(db_host)
    db = client[db_name]
    blocks_extracted = db['blocks_fragments']

    client = daisy.Client()

    while True:

        with client.acquire_block() as block:

            if block is None:
                break

            start = time.time()
            logging.info('Starting WATERSHED')

            try:
                args['block'] = block
                watershed_fun(**args)
            except Exception as e:
                block.status = BlockStatus.FAILED
                raise(e)

            document = {
                'block_id': block.block_id,
                'read_roi': (
                    block.read_roi.get_begin(),
                    block.read_roi.get_shape()
                ),
                'write_roi': (
                    block.write_roi.get_begin(),
                    block.write_roi.get_shape()
                ),
                'start': start,
                'duration': time.time() - start
            }

            blocks_extracted.insert_one(document)


if __name__ == '__main__':

    extract_fragments_worker(sys.argv[1])
