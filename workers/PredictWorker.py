from __future__ import print_function

import json
import logging
import pymongo
import sys

from funlib.learn.torch.models import UNet, ConvPass
from gunpowder import *
from gunpowder.torch import *
from torch.nn import Module


logging.getLogger('gunpowder.nodes.zarr_write').setLevel(logging.DEBUG)
logging.getLogger('gunpowder.profiling').setLevel(logging.INFO)


class AffsLsdModel(Module):

    def __init__(self, num_fmaps):

        super().__init__()

        self.unet = UNet(
                         in_channels=1,
                         num_fmaps=num_fmaps,
                         fmap_inc_factor=5,
                         downsample_factors=[
                                             [1, 2, 2],
                                             [1, 2, 2],
                                             [1, 2, 2]],
                         kernel_size_down=[
                                           [[3, 3, 3], [3, 3, 3]],
                                           [[3, 3, 3], [3, 3, 3]],
                                           [[3, 3, 3], [3, 3, 3]],
                                           [[3, 3, 3], [3, 3, 3]]],
                         kernel_size_up=[
                                         [[3, 3, 3], [3, 3, 3]],
                                         [[3, 3, 3], [3, 3, 3]],
                                         [[3, 3, 3], [3, 3, 3]]])

        self.conv_affs = ConvPass(num_fmaps, 3, [[1, 1, 1]], activation='Sigmoid')
        self.conv_lsds = ConvPass(num_fmaps, 10, [[1, 1, 1]], activation='Sigmoid')

    def forward(self, input):

        y = self.unet(input)
        affs = self.conv_affs(y)
        lsds = self.conv_lsds(y)

        return affs, lsds


def block_done_callback(
        db_host,
        db_name,
        block,
        start,
        duration):

    print("Recording block-done for %s" % (block,))

    client = pymongo.MongoClient(db_host)
    db = client[db_name]
    collection = db['blocks_predicted']

    document = {
        'block_id': block.block_id,
        'read_roi': (block.read_roi.get_begin(), block.read_roi.get_shape()),
        'write_roi': (block.write_roi.get_begin(), block.write_roi.get_shape()),
        'start': start,
        'duration': duration
    }

    collection.insert_one(document)

    print("Recorded block-done for %s" % (block,))

    
def predict(
        model_path,
        num_fmaps,
        raw_path,
        raw_dataset,
        output_path,
        output_dataset,
        input_size,
        output_size,
        db_host,
        db_name,
        num_cache_workers):

    model = AffsLsdModel(num_fmaps=num_fmaps)
    model.eval()
    
    raw = ArrayKey('RAW')
    lsds = ArrayKey('LSDS')
    affs = ArrayKey('AFFS')

    chunk_request = BatchRequest()
    chunk_request.add(raw, input_size)
    chunk_request.add(lsds, output_size)
    chunk_request.add(affs, output_size)

    pipeline = ZarrSource(
            store=raw_path,
            datasets = {raw: raw_dataset},
            array_specs = {raw: ArraySpec(interpolatable=True)}
        )

    pipeline += Pad(raw, size=None)

    pipeline += Normalize(raw)
    
    pipeline += Unsqueeze([raw])
    pipeline += Unsqueeze([raw])

    pipeline += Predict(
            model=model,
            checkpoint=model_path,
            inputs={
                'input': raw
            },
            outputs={
                0: affs,
                1: lsds
            },
            spawn_subprocess = True
        )

    #pipeline += IntensityScaleShift(affs, 255, 0)
    pipeline += Squeeze([affs])
    pipeline += ZarrWrite(
            dataset_names={
                # lsds: 'pred_lsds',
                affs: output_dataset
            },
            store=output_path
                         )

    pipeline += PrintProfilingStats(every=10)

    pipeline += DaisyRequestBlocks(
            chunk_request,
            roi_map={
                raw: 'read_roi',
                affs: 'write_roi',
                lsds: 'write_roi'
            },
            num_workers=num_cache_workers, # Workers queuing request batches for the GPU
            block_done_callback=lambda b, s, d: block_done_callback(
                db_host,
                db_name,
                b, s, d)
                )

    print("Starting prediction...")
    with build(pipeline):
        pipeline.request_batch(BatchRequest())
    print("Prediction finished")

if __name__ == "__main__":

    logging.basicConfig(level=logging.DEBUG)
    

    config_file = sys.argv[1]
    with open(config_file, 'r') as f:
        run_config = json.load(f)

    predict(**run_config)
