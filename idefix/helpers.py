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
import rasterio as rio
import sap
import higra as hg
from pathlib import Path

from .vxl import get_grid, bin, squash, fit_bbox
from .io import load_pc

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

def dtm_dh_filter(dsm, sigma=.5, epsilon=20000, alpha=2):
    """Compute a digital terrain model (DTM) from a DSM.

    Work best with DSM of last echo.

    Parameters
    ----------
    dsm : ndarray
        The DSM.
    sigma : scalar
        The height theshold to trigger object detection. Default is
        0.5 m.
    epsilon : scalar
        The area theshold for ground objects. All objects with surface
        greater than epsilon are forcedto be ground. Default is 20 km².
    alpha : scalar
        The area threshold for horizontal noise filter. Area variations
        smaller than alpha are removed for the detection of height
        threshold sigma. Default is 2 m².

    Returns
    -------
    dtm : ndarray
        The DTM computed from the DSM.

    """
    mt = sap.MaxTree(dsm)
    area = mt.get_attribute('area')

    area_child = hg.accumulate_parallel(mt._tree, area, hg.Accumulators.max)
    pruned = (area - area_child) <= alpha

    pruned_tree, pruned_map = hg.simplify_tree(mt._tree, pruned)

    dh = mt._alt[pruned_map] - mt._alt[pruned_map][pruned_tree.parents()]
    remove = dh > sigma

    original_map = np.zeros(mt.num_nodes(), dtype=np.int)
    original_map[pruned_map] = np.arange(pruned_map.size)
    original_map = hg.accumulate_and_max_sequential(mt._tree, original_map, np.arange(mt._tree.num_leaves()), hg.Accumulators.max)
    original_remove = remove[original_map] & (area < epsilon)

    dtm = mt.reconstruct(original_remove, filtering='min')

    return dtm

def write_raster(raster, bbox, crs, fname):
    """Write a GeoTiff
    """
    west, east = bbox[:,0]
    south, north = bbox[:,1]
    width, height = raster.shape

    west, east, south, north, width, height

    transform = rio.transform.from_bounds(west, south, east, north, width, height)
    dtype = raster.dtype

    with rio.open(fname,
                     mode='w',
                     driver='GTiff',
                     width=raster.shape[0],
                     height=raster.shape[1],
                     count=1,
                     crs=crs,
                     dtype=dtype,
                     transform=transform,
                     compress='lzw') as out_tif:
        out_tif.write(raster[None])

def rasterize(pc_file, feature='elevation', resolution=1.,
              bin_structure='voxel', bin_method='mean',
              squash_method='top', interpolation='idw',
              out_dir=None, crs=None):
    """Rasterize a point cloud file.

    Parameters
    ----------
    pc_file : Path
        Path of the point cloud file.
    feature : str
        'elevation' or LiDAR feature name contained in the point cloud
        file.
    resolution : float
        The spatial resolution of the raster.
    bin_structure : str
        'voxel' or 'pixel'.
    bin_method : str
        'mean', 'density' or 'mode'.
    squash_method : str or tuple of str
        'top', 'center', 'bottom', 'min', 'max', 'mean', 'median', 'std'
    interpolation : str
        'idw', 'nearest', 'linear', 'cubic'
    out_dir : Path or str or None
        Path of the directory for the result files. Optional.
    crs : str or None
        The coordinate reference system of the point cloud. Required if
        out_dir is defined to write the GeoTiff.

    Returns
    -------
    raster : ndarray
        Raster.
    """
    pc = load_pc(pc_file)

    steps = resolution if bin_structure == 'voxel' else (resolution, resolution, None)

    bbox = fit_bbox(pc.spatial)
    grid = get_grid(bbox, steps)
    #ix.vxl.insight(grid, method=bin_method, verbose=True)

    fval = pc.spatial[:,2] if feature == 'elevation' else getattr(pc.feature, feature)

    vxl = bin(grid, pc.spatial, fval, bin_method)

    squash_method = squash_method if not isinstance(squash_method, str) else (squash_method,)

    rasters = []
    for s in squash_method:
        raster = squash(vxl, s)
        raster = interpolate(raster, interpolation)
        if out_dir:
            format_dict = {'name': Path(pc_file).stem,
                           'feature': feature,
                           'resolution': resolution,
                           'bin_structure': bin_structure,
                           'bin_method': bin_method,
                           'squash_method': s,
                           'interpolation': interpolation}
            out_tif_name = Path(out_dir) / \
            '{name}_{feature}_{resolution}_{bin_structure}_{bin_method}_{squash_method}_{interpolation}.tif'.format(**format_dict)
            write_raster(raster, bbox, crs, out_tif_name)
        rasters += [raster]

    return np.array(rasters)
