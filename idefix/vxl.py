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

def _ui_step(step):
    '''User input management for step (number or array)
    '''
    try:
        iter(step)
        if len(step) != 3:
            msg = 'Wrong steps input, 3 steps expected in step = \'{}\''.format(step)
            log.error(msg)
            raise IOError(msg)
    except TypeError:
        step = [step] * 3
    return step

def get_grid(spatial, step):
    '''Return grid bins.

    Compute the grid bins of a spatial point cloud or corresponding bounding
    box according to given step (or steps for anisotropic grid).

    Parameters
    ----------
    spatial : array (n, 3)
        The spatial point cloud or the corresponding bounding box to grid.
    step : number or array or tuple
        The step of the grid, can be a number to get an isotropic grid, or an
        iterable of size 3 (required) to get an anisotropic grid.

    Returns
    -------
    grid : array of array (3,)
        Grid of spatial given step. Return three arrays (not necessarily of the
        same size) defining the bins of axis `x`, `y` and `z`.
    '''
    bb = bbox(spatial)
    step = _ui_step(step)

    #ipdb.set_trace()
    grid = []
    for start, stop, s in zip(bb[0], bb[1], step):
        grid += [np.arange(start, stop + 2*s, s)]

    return grid

