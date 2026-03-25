# EMsegment

A Python package for 3D electron microscopy (EM) image segmentation using deep learning. Implements a distributed, block-wise processing pipeline that segments volumetric EM data through three stages: prediction, fragment extraction, and agglomeration.

This project is based on [scripts](https://github.com/funkelab/lsd/tree/master/lsd/tutorial/scripts) by the Funke lab. Also see Acknowledgements section.

## Installation

First, clone the repository locally
```bash
git clone https://github.com/Heinze-lab/EMsegment.git
```

Create a new environment and activate it (here using conda)
```bash
conda create -n myenv python=3.12
conda activate myenv
```

Install dependencies and package
```bash
pip install -r requirements.txt
pip install -e .
```

Alternatively, create the environment using the provided environment.yml directly
```bash
conda env create --n myenv --file=environment.yml
```

_Installation was tested on Ubuntu 20.04 with Python 3.12._

### Requirements

- Python 3.12
- CUDA-capable GPU(s) for prediction
- MongoDB instance for progress tracking

## Quick Start

### Full Pipeline

```bash
python emsegment/Segment.py \
  -p /path/to/project_dir \
  -prefix my_project \
  -i /path/to/input.zarr \
  -m emsegment/config/model_config.json \
  -c 8 \
  --GPU 0 1 \
  --seg-config emsegment/seg_config.json \
  --db-host mongodb://localhost:27017
```

### Key Parameters

| Parameter | Description |
|-----------|-------------|
| `--GPU` | CUDA device ID(s) for prediction |
| `-c` | Number of CPU workers for fragments/agglomeration |
| `--chunk-voxel-size` | Block size [Z,Y,X] in voxels. Default: `[100,500,500]` |
| `--roi-start/--roi-size` | Process subset of a volume (in world unit) |
| `--continue-previous` | Resume interrupted job using the highest ID |
| `--start-over` | Clear progress and restart |
| `--todo` | Stages of the pipeline to go through (`predict`, `fragment`, `agglomerate`). Default: all stages |

For a description of all parameters, run `--help`.

## Configuration

### Model Configuration (`model_config.json`)

The model configuration contains information necessary to build the model and load a trained state.

```json
{
  "model_path": "/path/to/trained_model.pt",
  "num_fmaps": 12,
  "output_shape": [40, 200, 200],
  "padding": [20, 100, 100]
}
```

### Segmentation Configuration (`seg_config.json`)

The segmentation configuration contains parameters used for each stage of the segmentation pipeline.

```json
{
  "pred_config": {
    "models_per_gpu": 1,
    "num_cache_workers": 4,
    "write_affs": true,
    "write_lsds": false
  },
  "frag_config": {
    "context_px": [10, 50, 50],
    "fragments_in_xy": true,
    "epsilon_agglomerate": 0,
    "filter_fragments": 0,
    "min_seed_distance": 5
  },
  "agglo_config": {
    "threshold": 0.5,
    "edges_collection_basename": "edges",
    "merge_function": "hist_quant_25"
  }
}
```

## Data Format

- **Input**: Zarr containers with raw EM data
- **Predictions**: 4D arrays (channels, z, y, x) - 3 channels for affinities, 10 for LSDs
- **Fragments**: 3D uint64 label arrays
- **Agglomeration**: Weighted edges stored in MongoDB

## Acknowledgments

Built on tools from the [Funke Lab](https://github.com/funkelab):
- [daisy](https://github.com/funkelab/daisy) - Distributed processing
- [gunpowder](https://github.com/funkelab/gunpowder) - Data augmentation
- [lsd](https://github.com/funkelab/lsd) - Local shape descriptors

## License

MIT License
