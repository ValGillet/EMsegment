# EMsegment

A Python package for 3D electron microscopy (EM) image segmentation using deep learning. Implements a distributed, block-wise processing pipeline that segments volumetric EM data through three stages: prediction, fragment extraction, and agglomeration.

## Overview

EMsegment processes large EM volumes by:

1. **Prediction**: Running trained neural networks to generate affinity maps and/or local shape descriptors (LSDs)
2. **Fragments**: Applying watershed segmentation to create supervoxels from predictions
3. **Agglomeration**: Merging fragments into final segments using a region adjacency graph (RAG)

Each stage uses [daisy](https://github.com/funkelab/daisy) for distributed block-wise processing with MongoDB for progress tracking, enabling fault-tolerant processing of arbitrarily large volumes.

## Installation

```bash
pip install -r requirements.txt
pip install -e .
```

### Requirements

- Python 3.8+
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

### Individual Stages

```bash
# Prediction only
python emsegment/Segment.py ... --todo predict

# Fragments only (requires predictions)
python emsegment/Segment.py ... --todo fragment

# Agglomeration only (requires fragments)
python emsegment/Segment.py ... --todo agglomerate
```

### Extract Final Segments

After agglomeration, extract segments at different thresholds:

```bash
python emsegment/FindSegments.py config.json
```

## Configuration

### Model Configuration (`model_config.json`)

```json
{
  "model_path": "/path/to/trained_model.pt",
  "num_fmaps": 12,
  "output_shape": [40, 200, 200],
  "padding": [20, 100, 100]
}
```

### Segmentation Configuration (`seg_config.json`)

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

## Key Parameters

| Parameter | Description |
|-----------|-------------|
| `--GPU` | CUDA device IDs for prediction |
| `-c` | Number of CPU workers for fragments/agglomeration |
| `--chunk-voxel-size` | Block size [Z,Y,X] in voxels (default: 100,500,500) |
| `--roi-start/--roi-size` | Process subset of volume (in nm) |
| `--continue-previous` | Resume interrupted job |
| `--start-over` | Clear progress and restart |

## Data Format

- **Input**: Zarr containers with raw EM data
- **Predictions**: 4D arrays (channels, z, y, x) - 3 channels for affinities, 10 for LSDs
- **Fragments**: 3D uint64 label arrays
- **Output**: Lookup tables (LUTs) mapping fragment IDs to segment IDs

## Project Structure

```
emsegment/
├── Segment.py              # Main entry point
├── PredictBlockwise.py     # Prediction stage
├── FragmentsBlockwise.py   # Fragment extraction stage
├── AgglomerateBlockwise.py # Agglomeration stage
├── FindSegments.py         # Extract final segments
├── workers/                # Worker subprocess scripts
├── utils/                  # Utility functions
└── config/                 # Example configurations
```

## Acknowledgments

Built on tools from the [Funke Lab](https://github.com/funkelab):
- [daisy](https://github.com/funkelab/daisy) - Distributed processing
- [gunpowder](https://github.com/funkelab/gunpowder) - Data augmentation
- [lsd](https://github.com/funkelab/lsd) - Local shape descriptors

## License

MIT License
