#!/usr/bin/env python
# file test_vxl.py
# author Florent Guiotte <florent.guiotte@uhb.fr>
# version 0.0
# date 18 mars 2019
""" Test functions for the voxels `vxl` subpackage.

"""

from dataclasses import dataclass
import pytest
import numpy as np
from idefix import vxl

@dataclass
class Pcloud:
    spatial: np.ndarray
    feature: np.ndarray

def data_pc(datadir, set_id):
    path = datadir.join('pc{}.txt'.format(set_id))
    data = np.loadtxt(path)
    return Pcloud(data[:,:3], data[:,3])

#def data_vxl(datadir, set_id, step, method):
#    pass

@pytest.fixture
def data_0_vxl():
    def _data_0_vxl(method, resolution):
        if method == 'mean':
            pass

def data_grid(datadir, set_id, step_id):
    def _read(fname):
        with open(fname, 'r') as f:
            grid = []
            for arr in f.readlines():
                grid += [np.array([float(x) for x in arr.split(' ')])]
        return grid

    path = datadir.join('pc{}_grid_s{}.txt'.format(set_id, step_id))
    return _read(path)
    
@pytest.mark.parametrize('set_id, step, grid_id', [
    ('0', 1., '1'),
    ('0', .1, '0_1'),
])
def test_get_grid(datadir, set_id, step, grid_id):
    spatial = data_pc(datadir, set_id).spatial
    res = data_grid(datadir, set_id, grid_id)

    assert spatial is not None, 'Test data empty, test function is broken!'
    assert spatial.shape[-1] == 3, 'Test data malformed, test function is broken!'
    assert res is not None, 'Test data empty, test function is broken!'
    assert len(res) == 3, 'Test data malformed, test function is broken!'

    test = vxl.get_grid(spatial, step)

    assert test is not None, 'Function did not return anything :('
    assert len(test) == 3, 'Function doesn\'t give right number of axis'

    for axis_test, axis_truth in zip(test, res):
        assert axis_test.size == axis_truth.size, 'Wrong size for axis'
        assert (axis_test == axis_truth).all(), 'Axis inequality between truth and test'
                         
def test_grid():
    """
    - dtype
    - method
    - mask
    - data
    """
    pass

