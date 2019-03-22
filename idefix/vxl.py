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
from .utils import bbox
import ipdb

log = logging.getLogger(__name__)

def _ui_step(step, spatial):
    '''User input management for step (number or array)
    '''
    try:
        iter(step)
        if len(step) != spatial.shape[-1]:
            msg = 'Missmatch between steps count and spatial dimensions, {} step(s) expected while step = \'{}\''.format(spatial.shape[-1], step)
            log.error(msg)
            raise ValueError(msg)
        out_step = step
    except TypeError:
        out_step = [step] * spatial.shape[-1]

    for s in out_step:
        if s <= 0:
            msg = 'Step should be greater than 0, steps = \'{}\''.format(step)
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
        return _bin_density(grid, spatial)
    else:
        if feature is None:
            msg = 'Missing required argument : \'feature\''
            log.error(msg)
            raise ValueError(msg)
    if method == 'mean':
        return _bin_mean(grid, spatial, feature)
    if method == 'mode':
        return _bin_mode(grid, spatial, feature)

    msg = 'Method \'{}\' does not exist.'.format(method)
    log.error(msg)
    raise NotImplementedError(msg)

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
        del score, winner

    return np.ma.masked_array(max_value, max_score == 0)

