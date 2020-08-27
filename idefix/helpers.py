#!/usr/bin/env python
# file helpers.py
# author Florent Guiotte <florent.guiotte@irisa.fr>
# version 0.0
# date 24 août 2020
"""High-level helper functions.

This module contains high-level helper functions. This module shows many
exemple on the use of idefix package and other packages (sap, rasterio,
...) to process point clouds.

"""

import numpy as np
from scipy.interpolate import griddata
from rasterio import fill
import sap

from .vxl import get_grid, bin, squash

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

def dsm(pcloud, cell_size=1., last=False):
    """Create the digital surface model (DSM) of the point cloud.

    Parameters
    ----------
    pcloud : recarray
        A point cloud loaded with :mod:`idefix.io`.
    cell_size : scalar
        The size of the cells in meter. Cells are square. Default is 1
        meter.
    last : bool
        Specifies whether the first echo (`False`) or the last echo
        (`True`) should be taken into account. Default is `False`.

    Returns
    -------
    dsm : ndarray
        The DSM of the point cloud.

    """
    grid = get_grid(pcloud.spatial, cell_size)
    vxlg = bin(grid, pcloud.spatial, pcloud.spatial[:,2], 'mean')
    rstr = squash(vxlg, 'bottom' if last else 'top')
    rstr = interpolate(rstr, 'idw')

    return rstr
