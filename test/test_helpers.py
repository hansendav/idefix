#!/usr/bin/env python
# file test_helpers.py
# author Florent Guiotte <florent.guiotte@irisa.fr>
# version 0.0
# date 24 août 2020
"""Abstract

doc.
"""

import numpy as np
import pytest
from idefix import helpers, io

@pytest.fixture
def ma_raster():
    rs = np.random.RandomState(42)
    raster = rs.random((10,10))
    raster = np.ma.array(raster, mask=raster<.1)
    return raster

@pytest.mark.parametrize('method', 
        ['nearest', 'linear', 'cubic', 'idw'])
def test_interpolate(ma_raster, method):
    helpers.interpolate(ma_raster, method)

def _data_pc(datadir, set_id):
    path = datadir.join('pc{}.txt'.format(set_id))
    data = io.load_txt(path, 'x y z i'.split())
    return data

@pytest.mark.parametrize('params', [
    {},
    {'cell_size': 2.},
    {'last': True}])
def test_dsm(datadir, params):
    pc = _data_pc(datadir, 0)
    dsm = helpers.dsm(pc, **params)

    assert dsm is not None, 'Did not return anything...'
    assert not np.isnan(dsm).any(), 'Some missing values in DSM'
