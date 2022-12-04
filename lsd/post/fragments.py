import numpy as np
import waterz
from scipy.ndimage import label, \
        maximum_filter, \
        gaussian_filter, \
        distance_transform_edt
from skimage.segmentation import watershed
from skimage.filters import sobel, roberts, prewitt, threshold_otsu
from skimage.morphology import remove_small_holes, binary_dilation, disk

def watershed_from_lsds(
        lsds,
        background_mask=False,
        mode='prewitt',
        remove_holes=10,
        erode=None,
        min_seed_distance=10,
        return_seeds=False,
        return_distances=False):
    '''Extract initial fragments from local shape descripotes ``lsds`` using a 
    watershed transform. This assumes first three components are mean_offsets.'''

    fragments = np.zeros(lsds.shape[1:], dtype=np.uint64)
    boundary_distances = np.zeros(fragments.shape)
    depth = fragments.shape[0]

    if return_seeds:
        seeds = np.zeros(fragments.shape, dtype=np.uint64)

    id_offset = 0

    for z in range(depth):

        if depth == 1:
            dims = slice(2,4)
        else:
            dims = slice(4,6)

        boundary_distances[z] = np.sum(lsds[dims,z], axis=0) + lsds[-1,z]

        if mode == "sobel":
            boundary_distances[z] = sobel(boundary_distances[z])
        elif mode == "prewitt":
            boundary_distances[z] = prewitt(boundary_distances[z])
        elif mode == "roberts":
            boundary_distances[z] = roberts(boundary_distances[z])
        else:
            boundary_distances[z] *= -1

        boundary_distances[z] = gaussian_filter(boundary_distances[z], 1)
        thresh = threshold_otsu(boundary_distances[z])
        boundary_mask = boundary_distances[z] <= thresh/1.5

        if remove_holes is not None:
            boundary_mask = remove_small_holes(boundary_mask,remove_holes)

        if erode is not None:
            boundary_mask = binary_dilation(boundary_mask,disk(erode))

        boundary_distances[z] = distance_transform_edt(boundary_mask)

        if background_mask == False:
            boundary_mask = None

        ret = watershed_from_boundary_distance(
                boundary_distances[z],
                boundary_mask,
                return_seeds=return_seeds,
                min_seed_distance=min_seed_distance)

        fragments[z] = ret[0]

        if return_seeds:
            seeds[z] = ret[2]

        id_offset = ret[1]

    ret = (fragments,id_offset)

    if return_distances:
        ret += (boundary_distances,)

    return ret


def watershed_from_affinities(
        affs,
        background_mask=False,
        fragments_in_xy=False,
        return_seeds=False,
        min_seed_distance=10):
    '''Extract initial fragments from affinities using a watershed
    transform. Returns the fragments and the maximal ID in it.

    Returns:

        (fragments, max_id)
        or
        (fragments, max_id, seeds) if return_seeds == True'''

    if fragments_in_xy:

        mean_affs = 0.5*(affs[1] + affs[2])
        depth = mean_affs.shape[0]

        fragments = np.zeros(mean_affs.shape, dtype=np.uint64)
        if return_seeds:
            seeds = np.zeros(mean_affs.shape, dtype=np.uint64)

        id_offset = 0
        for z in range(depth):

            thresh = threshold_otsu(mean_affs[z])

            boundary_mask = mean_affs[z] >= thresh
            boundary_distances = distance_transform_edt(boundary_mask)

            if background_mask == False:
                boundary_mask = None

            ret = watershed_from_boundary_distance(
                boundary_distances,
                boundary_mask,
                return_seeds=return_seeds,
                id_offset=id_offset,
                min_seed_distance=min_seed_distance)

            fragments[z] = ret[0]
            if return_seeds:
                seeds[z] = ret[2]

            id_offset = ret[1]

        ret = (fragments, id_offset)
        if return_seeds:
            ret += (seeds,)

    else:

        thresh = threshold_otsu(np.mean(affs,axis=0))

        boundary_mask = np.mean(affs, axis=0) >= thresh
        boundary_distances = distance_transform_edt(boundary_mask)

        if background_mask == False:
            boundary_mask = None

        ret = watershed_from_boundary_distance(
            boundary_distances,
            boundary_mask,
            return_seeds,
            min_seed_distance=min_seed_distance)

        fragments = ret[0]

    return ret


def watershed_from_boundary_distance(
        boundary_distances,
        boundary_mask,
        return_seeds=False,
        id_offset=0,
        min_seed_distance=10):

    max_filtered = maximum_filter(boundary_distances, min_seed_distance)
    maxima = max_filtered==boundary_distances
    seeds, n = label(maxima)

    if n == 0:
        return np.zeros(boundary_distances.shape, dtype=np.uint64), id_offset

    seeds[seeds!=0] += id_offset

    fragments = watershed(
        boundary_distances.max() - boundary_distances,
        seeds,
        mask=boundary_mask)

    ret = (fragments.astype(np.uint64), n + id_offset)
    if return_seeds:
        ret = ret + (seeds.astype(np.uint64),)

    return ret
