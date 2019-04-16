#!/usr/bin/env python
# file io.py
# author Florent Guiotte <florent.guiotte@uhb.fr>
# version 0.0
# date 04 mars 2019
""" IO IDEFIX sub-package

General functions to load and dump data in various format.
"""

import logging
from pathlib import Path
import numpy as np
import numpy.core.records as rcd
from numpy.lib import recfunctions as rfn
import laspy

log = logging.getLogger(__name__)

def load_las(fname):
    '''Load a LAS file into idefix point cloud format.

    Notes
    -----
    The empty fields from LAS format will be automatically stripped. The fields
    typing are automatically determined.

    Parameters
    ----------
    fname : str, Path
        Path to the LAS file to load.

    Returns
    -------
    pcloud : recarray
        Point cloud respecting structure::

        [(spatial), (feature, [f1, f2, ..., fn])]
    '''
    fname = _get_verify_path(fname)

    log.info('Loading LAS file \'{}\'...'.format(fname))
    try:
        infile = laspy.file.File(fname)
    except Exception as e:
        msg = 'Laspy exception while opening file \'{}\': {}'.format(fname, e)
        log.error(msg)
        raise IOError(msg)

    log.debug('Extract spatial data')
    spatial = np.core.records.fromarrays([np.array((infile.x, infile.y, infile.z)).T],
                                         dtype=[('spatial', np.float, 3)])

    log.debug('Extract feature data')
    feature_data, feature_dtype = [], []
    for spec in infile.reader.point_format:
        if spec.name in 'XYZ':
            continue
        att = infile.reader.get_dimension(spec.name)
        if (att == 0).all():
            log.info('Drop empty \'{}\' feature'.format(spec.name))
        else:
            log.info('Load \'{}\' feature'.format(spec.name))
            feature_data.append(att)
            feature_dtype.append((spec.name, att.dtype))

    log.debug('Create feature recarray')
    feature = np.core.records.fromarrays(feature_data, dtype=feature_dtype)
    del feature_data, feature_dtype

    log.debug('Concatenate pcloud')
    pcloud = rfn.append_fields(spatial, 'feature', feature, usemask=False, asrecarray=True)

    return pcloud


def load_txt(fname, header, delimiter=' ', dtype=None):
    '''Load a text file into idefix point cloud format.

    Read point cloud from text files (CSV like).

    Notes
    -----
    This reader needs the column header corresponding to the point cloud. There
    has to be `x`, `y` and `z` columns in the header.

    Parameters
    ----------
    fname : str, Path
        Path to the text file to load.
    header : array, tuple
        Names of the columns contained in the text point cloud file.
    delimiter : str, optional
        String used to separate values. The default is whitespace.
    dtype : array, tuple
        Data types of the columns contained in the file. This list must match
        the `header` parameter. Default is None, data type inferred is float.

    Returns
    -------
    pcloud : recarray
        Point cloud respecting structure::

        [(spatial), (feature, [f1, f2, ..., fn])]
    '''
    fname = _get_verify_path(fname)

    if dtype is not None:
        assert len(dtype) == len(header), 'dtype and header must be the same size'

    log.info('Loading TXT file \'{}\'...'.format(fname))
    try:
        log.debug('Loading the first lines of \'{}\'...'.format(fname))
        insight_txt = np.loadtxt(fname, delimiter=delimiter, max_rows=2)
    except Exception as e:
        msg = 'Numpy exception while opening file \'{}\': {}'.format(fname, e)
        log.error(msg)
        raise IOError(msg)

    # Compare header length and column count
    if insight_txt.shape[-1] != len(header):
        msg = 'Mismatch between header and file columns, count {} and count {}'.format(insight_txt.shape[-1], len(header))
        log.error(msg)
        raise IOError(msg)

    dtype = (np.float,) * len(header) if not dtype else dtype
    processed_dtype = [(x, y) for x, y in zip(header, dtype)]
    raw_txt = np.loadtxt(fname, delimiter=delimiter, dtype=processed_dtype)

    log.debug('Extract spatial data')
    spatial = np.core.records.fromarrays([np.array([raw_txt[x] for x in ('x', 'y', 'z')]).T],
                                         dtype=[('spatial', np.float, 3)])

    log.debug('Extract feature data')
    header_c = list(header)
    for i in ('x', 'y', 'z'):
        header_c.remove(i)

    if not header_c:
        return spatial

    log.debug('Create feature recarray')
    feature = raw_txt[header_c]

    log.debug('Concatenate pcloud')
    pcloud = rfn.append_fields(spatial, 'feature', feature, usemask=False, asrecarray=True)

    return pcloud

def _get_verify_path(fname):
    fname = Path(fname)
    if not fname.is_file():
        msg = 'No such file: \'{}\''.format(fname)
        log.error(msg)
        raise IOError(msg)
    return fname

def _arr_to_rec(arr):
    """Array to record array.

    Used for point clouds, should work for everything else tho...
    """
    arrays = []; dtypes = []
    for k in arr.dtype.fields.keys():
        arrays += [arr[k]]
        dtypes += [(k, arr.dtype[k])]
    return np.core.records.fromarrays(arrays, dtypes)

def load_pc(fname):
    """Load point cloud from file.
    
    Loader for point clouds containted in compatible '.npz' files. This "point
    cloud" format is based on NumPy files, with small overhead to manage record
    array and multispectral point clouds.

    Parameters
    ----------
    fname : str, Path
        Path to the point cloud file to load.

    Returns
    -------
    point_cloud : recarray or tuple of recarray
        The point cloud or tuple of point clouds (for multispectral point cloud
        files).
    """
    log.info('Loading point cloud file \'{}\')'.format(fname))

    fname = _get_verify_path(fname)

    archive = np.load(fname)
    if len(archive.files) == 1:
        return _arr_to_rec(archive[archive.files[0]])
    else:
        return tuple(_arr_to_rec(archive[arr]) for arr in archive.files)

def dump_pc(fname, point_cloud, compress=False):
    """Dump point cloud to file.
    
    Write a point cloud (or several point clouds) in a '.npz' files.

    Parameters
    ----------
    fname : str, Path
        Path to the point cloud file to create.
    point_cloud : recarray or tuple of recarray
        The point cloud (or the tuple of point clouds) to dump.
    compress : bool
        Enable compression of the dumped file. Default is False.
    """
    if hasattr(point_cloud, 'spatial'):
        point_cloud = (point_cloud, )

    if compress:
        np.savez_compressed(fname, *point_cloud)
    else:
        np.savez(fname, *point_cloud)
