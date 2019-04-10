#!/usr/bin/env python
# file vxl.py
# author Florent Guiotte <florent.guiotte@uhb.fr>
# version 0.0
# date 19 mars 2019
""" Voxel IDEFIX sub-package

General functions to transform point clouds to voxels compatible with numpy.
"""

import logging
import numpy as np
import humanize
from .utils import bbox
import ipdb

log = logging.getLogger(__name__)

def _ui_step(step, spatial):
    '''User input management for step (number or array)
    '''
    try:
        iter(step)
        if len(step) != spatial.shape[-1]:
            msg = 'Missmatch between steps count and spatial dimensions, {} step(s) expected while step = \'{}\'.'.format(spatial.shape[-1], step)
            log.error(msg)
            raise ValueError(msg)
        out_step = step
    except TypeError:
        out_step = [step] * spatial.shape[-1]

    for s in out_step:
        if s <= 0:
            msg = 'Step should be greater than 0, steps = \'{}\'.'.format(step)
            log.error(msg)
            raise ValueError(msg)
    return out_step

def get_grid(spatial, step):
    '''Return grid bins.

    Compute the grid bins of a point cloud or the corresponding bounding box
    according to given step (or steps for anisotropic grid).

    Parameters
    ----------
    spatial : array (m, n)
        The spatial point cloud or the corresponding bounding box to grid.
    step : number or array or tuple
        The step of the grid, can be a number to get an isotropic grid, or an
        iterable of size 3 (required) to get an anisotropic grid.

    Returns
    -------
    grid : array of array (n,)
        Grid of spatial given step. Return three arrays (not necessarily of the
        same size) defining the bins of axis `x`, `y` and `z`.
    '''
    spatial = np.array(spatial)
    bb = bbox(spatial)
    step = _ui_step(step, spatial)

    grid = []
    for a_min, a_max, a_s in zip(bb[0], bb[1], step):
        # Beware of float underflow
        bins = np.trunc((a_max - a_min) / a_s).astype(int) + 1
        grid += [np.linspace(a_min, a_min + bins * a_s, bins + 1)]

    return grid

def bin(grid, spatial, feature=None, method='density'):
    '''Bin spatial data in a grid.

    Return a voxel grid representing the binned point cloud defined by point
    positions in `spatial`. The point cloud can be valued with the `feature`
    attribute.

    Parameters
    ----------
    grid : array of array (n,)
        Grid to bin spatial data.
    spatial : array (m, n)
        Spatial position of the points in R^n.
    feature : array (m)
        Point feature to represent in the bins. If None, density method is
        mandatory. Default is None.
    method : str
        Method to synthetize the point features in the grid. If the method is
        density, then the feature values are ignored. Implemented methods are:
        - 'density': The density of point in each cell.
        - 'mean': The mean of feature value in each cell.
        - 'mode': The modal (most common) in each cell. Designed for labels on
          point cloud, can be long with rich spectral data. If there is an
          equal number of elements, then the smallest is returned.
        The default is 'density'.

    Returns
    -------
    binned_pc : masked array (i, j, k)
        The binned point cloud, "No data" are masked.
    '''
    log.info('Bining point cloud in grid...')

    if method == 'density':
        geo_rst = _bin_density(grid, spatial)
    else:
        if feature is None:
            msg = 'Missing required argument : \'feature\'.'
            log.error(msg)
            raise ValueError(msg)
        if method == 'mean':
            geo_rst = _bin_mean(grid, spatial, feature)
        elif method == 'mode':
            geo_rst = _bin_mode(grid, spatial, feature)
        else:
            msg = 'Method \'{}\' does not exist.'.format(method)
            log.error(msg)
            raise NotImplementedError(msg)
    return _geo_to_np_coordinate(geo_rst)

def _bin_density(grid, spatial):
    '''Bin spatial in a grid, density method.
    '''
    density, edge = np.histogramdd(spatial, grid)
    vxl = np.ma.masked_array(density, density == 0)
    return vxl

def _bin_mean(grid, spatial, feature):
    '''Bin spatial in a grid, mean method.
    '''
    density, edge = np.histogramdd(spatial, grid)
    weightd, edge = np.histogramdd(spatial, grid, weights=feature)
    mask = density == 0
    return np.ma.masked_array(np.divide(weightd, density, where=~mask), mask)

def _bin_mode(grid, spatial, feature):
    '''Bin spatial in a grid, mode method.

    This function aim for efficiency with ndarray but is linearly impacted by
    number of unique values in spatial.
    '''
    log.info('Mode binning...')
    values = np.unique(feature)

    if values.size > 10:
        log.warn('Mode called on data with {} unique values, processing may be long.'.format(values.size))

    # Init
    max_score = np.zeros([len(x) - 1 for x in grid])
    max_value = np.zeros_like(max_score, dtype=feature.dtype)
    for i, value in enumerate(values):
        log.info('Processing value {}/{}'.format(i + 0, values.size))
        mask = np.argwhere(feature == value).reshape(-1)
        score = _bin_density(grid, spatial[mask])
        winner = score > max_score
        max_score[winner] = score[winner]
        max_value[winner] = value
        del score, winner, mask

    return np.ma.masked_array(max_value, max_score == 0)

def _bin_insight(grid):
    '''Return the predicted number of cells contained in grid.
    '''
    return np.prod([x.size - 1 for x in grid])

def _bin_density_insight(grid, dtype=np.float):
    density = np.dtype(dtype).itemsize
    res_data = density
    res_mask = np.dtype(np.bool).itemsize
    return _bin_insight(grid) * (density + res_data + res_mask)

def _bin_mean_insight(grid, feature=None):
    density = np.dtype(np.float).itemsize
    weight = np.dtype(np.float).itemsize
    mask = np.dtype(np.bool).itemsize
    res_data = np.dtype(np.float).itemsize
    res_mask = np.dtype(np.float).itemsize
    return (density + weight + mask + res_data + res_mask) * _bin_insight(grid)

def _bin_mode_insight(grid, feature=None):
    max_score = np.dtype(np.float).itemsize
    max_value = np.dtype(np.float).itemsize
    score = np.dtype(np.float).itemsize
    winner = np.dtype(np.bool).itemsize
    res_data = np.dtype(np.float).itemsize
    res_mask = np.dtype(np.bool).itemsize
    return _bin_insight(grid) * (max_score + max_value + max(score + winner, res_data + res_mask))

def insight(grid, feature=None, method='density', mem_limit=None, verbose=False):
    '''Display memory usage of binning process.

    Display in the logs (INFO level) the predicted memory usage needed by the
    binning process. If `mem_limit` is set, then the method will throw an
    exception (MemoryError) if the prediction exceed the limit.

    Parameters
    ----------
    grid : array of array (n,)
        Grid to bin spatial data.
    feature : array (m)
        Point feature to represent in the bins. If None, default float values
        are assumed.
    method : str
        Method to synthetize the point features in the grid.
    mem_limit : number, str
        The limit allowed to further process the grid. If the insight
        prediction exceed this limit a MemoryError is raised. If the parameter
        is a string, it can be set with human readable memory size (e.g.
        '3GB'). The default is bytes.
    verbose : bool
        Display on stdout (besides logging) the insights. Default is False.

    Return
    ------
    mem_usage : number
        The future RAM usage required to further process the data binning.
    '''
    if mem_limit is not None:
        mem_limit = _human_to_bytes(mem_limit) if isinstance(mem_limit, str) else mem_limit

    if method == 'density':
        mem_usage = _bin_density_insight(grid)
    elif method == 'mean':
        mem_usage = _bin_mean_insight(grid, feature)
    elif method == 'mode':
        mem_usage = _bin_mode_insight(grid, feature)
    else:
        raise IOError('Wrong method: \'{}\''.format(method))

    lines = _print_insight(grid, mem_usage, mem_limit)

    for l in lines:
        log.info(l)

    if verbose:
        print('\n'.join(lines))

    if mem_limit and mem_usage > mem_limit:
        msg = 'The memory requirement is higher than allowed memory usage.'
        log.error(msg)
        raise MemoryError(msg)

    return mem_usage

def _print_insight(grid, mem_usage, mem_limit):
    print_lines = [
    '--- GRID INSIGHT ---',
    'Grid shape:     \t{}'.format([x.size - 1 for x in grid]),
    'Number of cells:\t{}'.format(humanize.intword(_bin_insight(grid))),
    'Predicted RAM usage:\t{}'.format(humanize.naturalsize(mem_usage, binary=True)),
    'Allowed max RAM usage:\t{}'.format(humanize.naturalsize(mem_limit, binary=True) if mem_limit else 'Not set'),
    '--------------------',]
    return print_lines

def _human_to_bytes(human_size):
    bytes_count = {'KB': 1, 'MB': 2, 'GB': 3}
    for k, v in bytes_count.items():
        if human_size.endswith(k):
            return float(human_size.strip(k)) * 1024 ** v
    raise IOError('Did not understand size: \'{}\''.format(human_size))

def _geo_to_np_coordinate(raster):
    '''Geographic to numpy coordinate system.

    Transfer the raster (2D and 3D) from a geographic coordinate system to the
    numpy coordinate system.
    '''
    return np.flip(np.swapaxes(raster, 0, 1), 0)

def _squash_position(voxel_grid, method, axis):
    squash_mask = np.zeros_like(voxel_grid, dtype=np.int)
    mask_idx = (~voxel_grid.mask).nonzero()
    squash_mask[mask_idx] = mask_idx[axis]

    if method == 'top':
        squash_id = squash_mask.max(axis=axis).astype(np.uint)
    elif method == 'center':
        squash_id = np.ma.median(squash_mask, axis=axis).astype(np.uint)
    elif method == 'bottom':
        squash_id = squash_mask.min(axis=axis).astype(np.uint)

    xy_where = np.nonzero(~squash_id.mask)
    voxel_grid_where = list(xy_where)
    voxel_grid_where.insert(axis%(len(voxel_grid_where)+1), squash_id.compressed())

    raster = np.zeros_like(squash_id)
    raster[xy_where] = voxel_grid[tuple(voxel_grid_where)]

    return raster

def squash(voxel_grid, method='top', axis=-1):
    """Flatten a voxel grid.

    Squash the voxel grid along `axis` according to `method` into a raster.

    The squash methods proposed are :

    - Position based in the "column" (i.e. along axis).
        + 'top': The first non empty cells (from top) is returned.
        + 'center': The most centered cell is returned.
        + 'bottom': The last
    - Cell description in the "column".
        + 'count': The number of non empty cells.
        + 'mean': The mean value of the non empty cells.
        + 'median': The median value...
        + 'std': ...
        + 'min': ...
        + 'max': ...

    Parameters
    ----------
    voxel_grid : masked array (3D)
        The voxel grid (binned point cloud) to squash.
    method : str
        The squash method. It can be 'top', 'center', 'bottom', 'count', 'min',
        'mean', 'max', 'std' or 'median'. Default is 'top'.
    axis : number
        The axis to squash along. Default is last (i.e. 2 for 3D voxel grid).

    Return
    ------
    raster_grid : masked array (2D)
        The squashed raster.
    """
    if method in ('top', 'center', 'bottom'):
        return _squash_position(voxel_grid, method, axis)
    elif method == 'count':
        return ~voxel_grid.mask.sum(axis=axis)
    elif method == 'mean':
        return voxel_grid.mean(axis=axis)
    elif method == 'median':
        return np.ma.median(voxel_grid, axis=axis)
    elif method == 'min':
        return voxel_grid.min(axis=axis)
    elif method == 'max':
        return voxel_grid.max(axis=axis)
    elif method == 'std':
        return voxel_grid.std(axis=axis)
    
    raise NotImplementedError('Method \'{}\' does not exist.'.format(method))

