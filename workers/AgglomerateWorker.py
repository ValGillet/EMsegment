import daisy
import json
import logging
import os
import pymongo
import sys
import time

from funlib.persistence import open_ds
from lsd.post.persistence.mongodb_rag_provider import MongoDbRagProvider
from lsd.post.parallel_aff_agglomerate import agglomerate_in_block

logging.basicConfig(level = logging.INFO)


def agglomerate_worker(input_config):

    logging.info(sys.argv)

    with open(input_config, 'r') as f:
        config = json.load(f)

    logging.info(config)

    affs_path = config['affs_path']
    affs_dataset = config['affs_dataset']
    fragments_path = config['fragments_path']
    fragments_dataset = config['fragments_dataset']
    db_host = config['db_host']
    db_name = config['db_name']
    edges_collection = config['edges_collection']
    merge_function_key = config['merge_function']
    threshold = config['threshold']

    merge_function = {
                'hist_quant_10': 'OneMinus<HistogramQuantileAffinity<RegionGraphType, 10, ScoreValue, 256, false>>',
                'hist_quant_10_initmax': 'OneMinus<HistogramQuantileAffinity<RegionGraphType, 10, ScoreValue, 256, true>>',
                'hist_quant_25': 'OneMinus<HistogramQuantileAffinity<RegionGraphType, 25, ScoreValue, 256, false>>',
                'hist_quant_25_initmax': 'OneMinus<HistogramQuantileAffinity<RegionGraphType, 25, ScoreValue, 256, true>>',
                'hist_quant_50': 'OneMinus<HistogramQuantileAffinity<RegionGraphType, 50, ScoreValue, 256, false>>',
                'hist_quant_50_initmax': 'OneMinus<HistogramQuantileAffinity<RegionGraphType, 50, ScoreValue, 256, true>>',
                'hist_quant_75': 'OneMinus<HistogramQuantileAffinity<RegionGraphType, 75, ScoreValue, 256, false>>',
                'hist_quant_75_initmax': 'OneMinus<HistogramQuantileAffinity<RegionGraphType, 75, ScoreValue, 256, true>>',
                'hist_quant_90': 'OneMinus<HistogramQuantileAffinity<RegionGraphType, 90, ScoreValue, 256, false>>',
                'hist_quant_90_initmax': 'OneMinus<HistogramQuantileAffinity<RegionGraphType, 90, ScoreValue, 256, true>>',
                'mean': 'OneMinus<MeanAffinity<RegionGraphType, ScoreValue>>',
                }[merge_function_key]
    
    logging.info(f'Reading affs from {affs_path}')
    affs = open_ds(os.path.join(affs_path, affs_dataset))

    logging.info(f'Reading fragments from {fragments_path}')
    fragments = open_ds(os.path.join(fragments_path, fragments_dataset))

    # Open RAG DB
    logging.info("Opening RAG DB...")
    rag_provider = MongoDbRagProvider(db_name,
                                      host=db_host,
                                      edges_collection = edges_collection,
                                      mode='r+')
    logging.info("RAG DB opened")

    # Open block done DB
    client = pymongo.MongoClient(db_host)
    db = client[db_name]
    blocks_agglomerated = db['blocks_agglomerated_' + merge_function_key]

    client = daisy.Client()

    while True:

        with client.acquire_block() as block:

            if block is None:
                break

            start = time.time()

            agglomerate_in_block(
                    affs,
                    fragments,
                    rag_provider,
                    block,
                    merge_function = merge_function,
                    threshold = threshold)

            document = {
                'block_id': block.block_id,
                'read_roi': (block.read_roi.get_begin(), block.read_roi.get_shape()),
                'write_roi': (block.write_roi.get_begin(), block.write_roi.get_shape()),
                'start': start,
                'duration': time.time() - start
            }

            blocks_agglomerated.insert_one(document)



if __name__ == '__main__':

    agglomerate_worker(sys.argv[1])
