from lsd.post.parallel_fragments import get_mask_data_in_roi
from lsd.post.fragments import watershed_from_affinities, watershed
from funlib.geometry import Coordinate
from funlib.persistence import Array
from funlib.segment.arrays import relabel, replace_values
from scipy.ndimage import center_of_mass
import logging
import numpy as np
import waterz

logger = logging.getLogger(__name__)

'''
Code extracted from lsd by Funke lab.
'''

def watershed_in_block_affs(
        affs,
        block,
        context,
        rag_provider,
        fragments_out,
        num_voxels_in_block,
        mask=None,
        fragments_in_xy=False,
        epsilon_agglomerate=0.0,
        filter_fragments=0.0,
        min_seed_distance=10,
        replace_sections=None):
    '''

    Args:

        filter_fragments (float):

            Filter fragments that have an average affinity lower than this
            value.

        min_seed_distance (int):

            Controls distance between seeds in the initial watershed. Reducing
            this value improves downsampled segmentation.
    '''

    total_roi = affs.roi
    voxel_size = affs.voxel_size
    read_roi = total_roi.intersect(block.read_roi)

    logger.debug("reading affs from %s", block.read_roi)
    logger.debug("intersect with total_roi: %s", read_roi)

    affs = affs[read_roi]

    if affs.dtype == np.uint8:
        logger.info("Assuming affinities are in [0,255]")
        max_affinity_value = 255.0
        affs = affs.astype(np.float32)
    else:
        max_affinity_value = 1.0

    if mask is not None:

        logger.debug("reading mask from %s", read_roi)
        mask_data = get_mask_data_in_roi(mask, read_roi, voxel_size)
        logger.debug("masking affinities")
        affs *= mask_data

    # extract fragments
    fragments_data, _ = watershed_from_affinities(
        affs,
        max_affinity_value,
        fragments_in_xy=fragments_in_xy,
        min_seed_distance=min_seed_distance)

    if mask is not None:
        fragments_data *= mask_data.astype(np.uint64)

    if filter_fragments > 0:

        if fragments_in_xy:
            average_affs = np.mean(affs[0:2]/max_affinity_value, axis=0)
        else:
            average_affs = np.mean(affs/max_affinity_value, axis=0)

        filtered_fragments = []

        fragment_ids = np.unique(fragments_data)

        for fragment, mean in zip(
                fragment_ids,
                mean(
                    average_affs,
                    fragments_data,
                    fragment_ids)):
            if mean < filter_fragments:
                filtered_fragments.append(fragment)

        filtered_fragments = np.array(
            filtered_fragments,
            dtype=fragments_data.dtype)
        replace = np.zeros_like(filtered_fragments)
        replace_values(fragments_data, filtered_fragments, replace, inplace=True)

    if epsilon_agglomerate > 0:

        logger.info(
            "Performing initial fragment agglomeration until %f",
            epsilon_agglomerate)

        generator = waterz.agglomerate(
                affs=affs/max_affinity_value,
                thresholds=[epsilon_agglomerate],
                fragments=fragments_data,
                scoring_function='OneMinus<HistogramQuantileAffinity<RegionGraphType, 25, ScoreValue, 256, false>>',
                discretize_queue=256,
                return_merge_history=False,
                return_region_graph=False)
        fragments_data[:] = next(generator)

        # cleanup generator
        for _ in generator:
            pass

    if replace_sections:

        logger.info("Replacing sections...")

        block_begin = block.write_roi.get_begin()
        shape = block.write_roi.get_shape()

        z_context = context[0]/voxel_size[0]
        logger.info("Z context: %i",z_context)

        mapping = {}

        voxel_offset = block_begin[0]/voxel_size[0]

        for i,j in zip(
                range(fragments_data.shape[0]),
                range(shape[0])):
            mapping[i] = i
            mapping[j] = int(voxel_offset + i) \
                    if block_begin[0] == total_roi.get_begin()[0] \
                    else int(voxel_offset + (i - z_context))

        logging.info('Mapping: %s', mapping)

        replace = [k for k,v in mapping.items() if v in replace_sections]

        for r in replace:
            logger.info("Replacing mapped section %i with zero", r)
            fragments_data[r] = 0

    fragments = Array(fragments_data, read_roi.get_begin(), voxel_size)

    # crop fragments to write_roi
    fragments = fragments[block.write_roi]
    max_id = fragments.max()

    # ensure we don't have IDs larger than the number of voxels (that would
    # break uniqueness of IDs below)
    if max_id > num_voxels_in_block:
        logger.warning(
            "fragments in %s have max ID %d, relabelling...",
            block.write_roi, max_id)
        fragments, max_id = relabel(fragments)

        assert max_id < num_voxels_in_block

    # ensure unique IDs
    id_bump = block.block_id[1]*num_voxels_in_block
    logger.debug("bumping fragment IDs by %i", id_bump)
    fragments[fragments>0] += id_bump
    fragment_ids = range(id_bump + 1, id_bump + 1 + int(max_id))

    # store fragments
    logger.debug("writing fragments to %s", block.write_roi)
    fragments_out[block.write_roi] = fragments

    # following only makes a difference if fragments were found
    if max_id == 0:
        return

    # get fragment centers
    fragment_centers = {
        fragment: block.write_roi.get_offset() + voxel_size*Coordinate(center)
        for fragment, center in zip(
            fragment_ids,
            center_of_mass(fragments, fragments, fragment_ids))
        if not np.isnan(center[0])
    }

    # store nodes
    rag = rag_provider[block.write_roi]
    rag.add_nodes_from([
        (node, {
            'center_z': c[0],
            'center_y': c[1],
            'center_x': c[2]
            }
        )
        for node, c in fragment_centers.items()
    ])
    rag.sync_nodes()



def watershed_in_block_lsds(
        lsds,
        sigma,
        block,
        rag_provider,
        fragments_out,
        num_voxels_in_block,
        mask=None):
    '''

    Args:

        filter_fragments (float):

            Filter fragments that have an average affinity lower than this
            value.

        min_seed_distance (int):

            Controls distance between seeds in the initial watershed. Reducing
            this value improves downsampled segmentation.
    '''

    total_roi = lsds.roi
    voxel_size = lsds.voxel_size
    read_roi = total_roi.intersect(block.read_roi)

    logger.debug("reading lsds from %s", block.read_roi)
    logger.debug("intersect with total_roi: %s", read_roi)

    lsds = lsds[read_roi]

    if mask is not None:

        logger.debug("reading mask from %s", read_roi)
        mask_data = get_mask_data_in_roi(mask, read_roi, voxel_size)
        logger.debug("masking affinities")
        lsds *= mask_data

    # extract fragments
    fragments_data, _ = watershed(
        lsds,
        sigma)

    if mask is not None:
        fragments_data *= mask_data.astype(np.uint64)

    fragments = Array(fragments_data, read_roi.get_begin(), voxel_size)

    # crop fragments to write_roi
    fragments = fragments[block.write_roi]
    max_id = fragments.max()

    # ensure we don't have IDs larger than the number of voxels (that would
    # break uniqueness of IDs below)
    if max_id > num_voxels_in_block:
        logger.warning(
            "fragments in %s have max ID %d, relabelling...",
            block.write_roi, max_id)
        fragments, max_id = relabel(fragments)

        assert max_id < num_voxels_in_block

    # ensure unique IDs
    id_bump = block.block_id[1]*num_voxels_in_block
    logger.debug("bumping fragment IDs by %i", id_bump)
    fragments[fragments>0] += id_bump
    fragment_ids = range(id_bump + 1, id_bump + 1 + int(max_id))

    # store fragments
    logger.debug("writing fragments to %s", block.write_roi)
    fragments_out[block.write_roi] = fragments

    # following only makes a difference if fragments were found
    if max_id == 0:
        return

    # get fragment centers
    fragment_centers = {
        fragment: block.write_roi.get_offset() + voxel_size*Coordinate(center)
        for fragment, center in zip(
            fragment_ids,
            center_of_mass(fragments, fragments, fragment_ids))
        if not np.isnan(center[0])
    }

    # store nodes
    rag = rag_provider[block.write_roi]
    rag.add_nodes_from([
        (node, {
            'center_z': c[0],
            'center_y': c[1],
            'center_x': c[2]
            }
        )
        for node, c in fragment_centers.items()
    ])
    rag.sync_nodes()