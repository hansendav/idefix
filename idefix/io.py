#!/usr/bin/env python
# file io.py
# author Florent Guiotte <florent.guiotte@uhb.fr>
# version 0.0
# date 04 mars 2019
"""Abstract

doc.
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
    fname : string, Path
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
    infile = laspy.file.File(fname)

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


