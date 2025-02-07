import daisy
import json
import logging
import os
import pymongo
import sys
import time

from daisy.block import BlockStatus
from funlib.persistence import open_ds
from lsd.post.persistence.mongodb_rag_provider import MongoDbRagProvider
from lsd.post.parallel_fragments import watershed_in_block


logging.basicConfig(level = logging.DEBUG)
logging.getLogger('pymongo').setLevel(logging.WARNING) # Hide pymongo output when debugging

def extract_fragments_worker(input_config):

    logging.info(sys.argv)

    with open(input_config, 'r') as f:
        config = json.load(f)

    logging.info(config)
    
    # Read config (obsolete if with main script)
    affs_file            = config['affs_file']
    affs_dataset         = config['affs_dataset']
    fragments_file       = config['fragments_file']
    fragments_dataset    = config['fragments_dataset']
    db_host              = config['db_host']
    db_name              = config['db_name']
    context              = config['context']
    num_voxels_in_block  = config['num_voxels_in_block']
    fragments_in_xy      = config['fragments_in_xy']
    epsilon_agglomerate  = config['epsilon_agglomerate']
    filter_fragments     = config['filter_fragments']
    replace_sections     = config['replace_sections']    
    min_seed_distance    = config['min_seed_distance']
    mask_file            = config['mask_file']
    mask_dataset         = config['mask_dataset']
    
    # Open files
    logging.info(f'Reading affs from {affs_file}')

    affs = open_ds(os.path.join(affs_file, affs_dataset), mode = 'r')
    
    logging.info(f'Reading fragments from {fragments_file}')

    fragments = open_ds(os.path.join(fragments_file, fragments_dataset), mode='r+')

    if mask_file and isinstance(mask_file, str):
        logging.info(f'Reading mask from {mask_file}')
        mask = open_ds(os.path.join(mask_file, mask_dataset))
    else:
        mask = None

    # Open RAG DB
    logging.info('Opening RAG DB...')
    rag_provider = MongoDbRagProvider(db_name,
                                      host=db_host,
                                      mode='r+')
    logging.info('RAG DB opened')

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
                watershed_in_block(
                    affs,
                    block,
                    context,
                    rag_provider,
                    fragments,
                    num_voxels_in_block=num_voxels_in_block,
                    fragments_in_xy=fragments_in_xy,
                    min_seed_distance=min_seed_distance,
                    mask=mask,
                    epsilon_agglomerate=epsilon_agglomerate,
                    filter_fragments=filter_fragments,
                    replace_sections=replace_sections
                    )
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
