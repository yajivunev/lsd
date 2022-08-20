import numpy as np
import logging
import waterz
from scipy.ndimage import label, \
        maximum_filter, \
        distance_transform_edt
from skimage.segmentation import watershed
from skimage.filters import sobel, roberts, prewitt, threshold_otsu
from skimage.restoration import denoise_tv_chambolle

logger = logging.getLogger(__name__)


def watershed_from_lsds(
        lsds,
        denoise_wt=0.05,
        background_mask=False, 
        mode='prewitt',
        min_seed_distance=10,
        return_seeds=False, 
        return_distances=False):
    '''Extract initial fragments from local shape descriptors ``lsds`` using a
    watershed transform. This assumes that the first three entries of
    ``lsds`` for each voxel are vectors pointing towards the center.'''
    
    fragments = np.zeros(lsds.shape[1:], dtype=np.uint64)
    boundary_distances = np.zeros(fragments.shape)
    depth = fragments.shape[0]
        
    if denoise_wt is not None:
        lsds = np.stack([denoise_tv_chambolle(x,weight=denoise_wt) for x in lsds])

    if return_seeds:
        seeds = np.zeros(fragments.shape, dtype=np.uint64)

    id_offset = 0

    for z in range(depth):
        
        if mode == "sobel":
            sob =  np.sum([sobel(x) for x in lsds[1:3,z]],axis=0)
        elif mode == "prewitt":
            sob =  np.sum([prewitt(x) for x in lsds[1:3,z]],axis=0)
        elif mode == "roberts":
            sob =  np.sum([roberts(x) for x in lsds[1:3,z]],axis=0)
        else: raise AssertionError("unknown watershed mode. choose 'sobel' or 'roberts' or 'prewitt'.")
        
        thresh = threshold_otsu(sob)
        boundary_mask = sob <= thresh
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
        denoise_wt=None,
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

    if denoise_wt is not None:
        affs = np.stack([denoise_tv_chambolle(x,weight=denoise_wt) for x in affs])

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
        
        boundary_mask = np.mean(affs[:3],axis=0) >= thresh
        boundary_distances = distance_transform_edt(boundary_mask)
 
        if background_mask == False:
            boundary_mask = None
        
        ret = watershed_from_boundary_distance(
            boundary_distances,
            boundary_mask,
            return_seeds=return_seeds,
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

    logger.info(f"Found {n} fragments")

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
