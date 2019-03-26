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
    fname = Path(fname)
    if not fname.is_file():
        msg = 'No such file: \'{}\''.format(fname)
        log.error(msg)
        raise IOError(msg)

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


def load_txt(fname, header, delimiter=' '):
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

    Returns
    -------
    pcloud : recarray
        Point cloud respecting structure::

        [(spatial), (feature, [f1, f2, ..., fn])]
    '''
    fname = Path(fname)
    if not fname.is_file():
        msg = 'No such file: \'{}\''.format(fname)
        log.error(msg)
        raise IOError(msg)

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

    dtype = [(x, np.float) for x in header]
    raw_txt = np.loadtxt(fname, delimiter=delimiter, dtype=dtype)

    log.debug('Extract spatial data')
    spatial = np.core.records.fromarrays([np.array([raw_txt[x] for x in ('x', 'y', 'z')]).T],
                                         dtype=[('spatial', np.float, 3)])

    log.debug('Extract feature data')
    header_c = list(header)
    for i in ('x', 'y', 'z'):
        header_c.remove(i)

    log.debug('Create feature recarray')
    feature = raw_txt[header_c]

    log.debug('Concatenate pcloud')
    pcloud = rfn.append_fields(spatial, 'feature', feature, usemask=False, asrecarray=True)

    return pcloud
