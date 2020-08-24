#!/usr/bin/env python
# file utils.py
# author Florent Guiotte <florent.guiotte@uhb.fr>
# version 0.0
# date 28 févr. 2019
"""General utils functions.

This module contains common utils for basic point cloud management and dataviz.

Notes
-----

Everything should be highly tested there.

"""

import numpy as np
import logging
from scipy.interpolate import griddata
from rasterio import fill

log = logging.getLogger(__name__)

def first(a):
    """Returns the inverse of the parameter.

    Just a basic function to test pytest and sphinx autodoc.

    Parameters
    ----------
    a : integer
        Value to process.

    Returns
    -------
    b : integer
        Inverse of a.
    """
    log.info('first called.')
    return -a

def bbox(data):
    """Returns bounding box of data.

    This function returns the lower and the upper points describing the
    bounding box of the points contained in data.

    Parameters
    ----------
    data : ndarray (N, 3)
        Point cloud data of shape (N, 3), i.e. (x,y,z).

    Returns
    -------
    bbox : ndarray
        Lower and upper points describing the bounding box such as::

        [[xmin, ymin, zmin],
         [xmax, ymax, zmax]]
    """
    return np.array((np.min(data, axis=0), np.max(data, axis=0)))


def interpolate(raster, method='linear'):
    """Interpolate masked raster.

    Wrapper function to interpolate missing values in masked raster.
    The 'linear', 'nearest' and 'cubic' implementation are from `Scipy`_
    while the 'idw' (inverse distance weighting) is provided by
    `rasterio`_.

    .. _Scipy: https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.griddata.html
    .. _rasterio: https://rasterio.readthedocs.io/en/latest/api/rasterio.fill.html

    Parameters
    ----------
    raster : masked ndarray
        The raster with missing values masked.
    method : str
        Can be 'linear', 'nearest', 'cubic' or 'idw'.

    Returns
    -------
    out : ndarray
        The raster with filled missing values.

    """
    if method == 'idw':
        raster = fill.fillnodata(raster)
    else:
        coords = np.argwhere(~raster.mask)
        values = raster.compressed()
        grid   = np.argwhere(raster.mask)

        raster[raster.mask] = griddata(coords, values, grid, method=method)

        if method != 'nearest':
            raster.mask = np.isnan(raster)
            raster = interpolate(raster, 'nearest')

        raster = np.array(raster)

    assert not np.isnan(raster).any()

    return raster
