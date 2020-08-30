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

def data_vxl(datadir, set_id, grid_id, method):
    def _unpack_vxl(spatial, feature):
        coords = tuple([spatial[:,i] for i in range(3)])

        vxld = np.zeros(spatial.max(axis=0) + 1)
        vxld[coords] = feature

        vxlm = np.ones_like(vxld, dtype=np.bool)
        vxlm[coords] = False

        return np.ma.masked_array(vxld, vxlm)

    def _load_vxl(fname, feature_name):
        fields = ('x', 'y', 'z', 'density', 'mean', 'mode')

        i = fields.index(feature_name)

        data = np.loadtxt(fname)
        spatial = data[:,:3].astype(np.intp)
        feature = data[:,i]

        return _unpack_vxl(spatial, feature)

    path = datadir.join('pc{}_vxl_s{}.txt'.format(set_id, grid_id))
    return _load_vxl(path, method)

def data_raster(datadir, set_id, grid_id, axis, method):
    def _load_raster(fname):
        data = np.loadtxt(fname)
        return np.ma.masked_array(data, data==0)

    path = datadir.join('pc{}_vxl_s{}_raster_{}_{}.txt'.format(set_id, grid_id, axis, method))
    return _load_raster(path)

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
    ('0', 2., '2'),
    ('0', .1, '0_1'),
    ('0', .6, '0_6'),
    ('0', .7, '0_7'),
    ('0', .15, '0_15'),
    ('0', [1.,1.,2.] , '1-1-2'),
    ('0', [1.,1.,None] , '1-1-n'),
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
        assert np.allclose(axis_test, axis_truth), 'Axis inequality between truth and test'
        #assert (axis_test - axis_truth == 0).all(), 'Float overflow in tested grid'

def test_get_grid_ui():
    np.random.seed(0)
    spatial_2D = np.random.random((100,2))
    spatial_3D = np.random.random((100,3))

    with pytest.raises(ValueError,) as e_info:
        vxl.get_grid(spatial_3D, -1), 'Negativ test'

    with pytest.raises(ValueError) as e_info:
        vxl.get_grid(spatial_3D, [1., -1., 1.])

    with pytest.raises(ValueError) as e_info:
        vxl.get_grid(spatial_3D, [1., 1.])

    with pytest.raises(ValueError) as e_info:
        vxl.get_grid(spatial_2D, [1., 1., 1.])

def test_bin_ui():
    spatial = np.random.random((10,3))
    feature = np.random.random((10))
    grid    = [np.arange(0,1,.1)] * 3

    with pytest.raises(ValueError) as e_info:
        vxl.bin(grid, spatial, method='mean')

    with pytest.raises(NotImplementedError) as e_info:
        vxl.bin(grid, spatial, feature, method='🍆')

@pytest.mark.parametrize('set_id, grid_id, method', [
    ('0', '1', 'density'),
    ('0', '1', 'mean'),
    ('0', '1', 'mode'),
])
def test_bin(datadir, grid_id, set_id, method):
    data = data_pc(datadir, set_id)
    grid = data_grid(datadir, set_id, grid_id)
    truth = data_vxl(datadir, set_id, grid_id, method)

    test = vxl.bin(grid, data.spatial, data.feature, method)

    assert test is not None, 'Tested function did not return anything :('
    assert hasattr(test, 'mask'), 'The array is not masked!'
    assert test.shape == tuple([x.size - 1 for x in grid]), 'Voxel grid shape and test grid missmatch'
    assert (test.mask == truth.mask).all(), 'The returned mask is different from test truth'
    assert np.allclose(test.compressed(), truth.compressed()), 'The returned values are different from test truth'

@pytest.mark.parametrize('set_id, grid_id, cells', [
    ('0', '1', 1000),
    ('0', '2', 125),
    ('0', '0_1', 753571),
    ('0', '0_15', 226981),
])
def test__bin_insight(datadir, set_id, grid_id, cells):
    grid = data_grid(datadir, set_id, grid_id)
    assert vxl._bin_insight(grid) is not None, 'Tested function did not return anything :('
    assert vxl._bin_insight(grid) == cells, 'Private insight function did not return the correct number of cells for grid'

@pytest.mark.parametrize('method', [
    ('density'), ('mean'), ('mode')])
def test_insight(method):
    # Create a huge grid
    grid = [np.arange(1, 10, .0001)] * 3
    with pytest.raises(MemoryError) as e_info:
        vxl.insight(grid, method=method, mem_limit='3GB')
    with pytest.raises(MemoryError) as e_info:
        vxl.insight(grid, method=method, mem_limit='300 MB')
    with pytest.raises(MemoryError) as e_info:
        vxl.insight(grid, method=method, mem_limit='3KB')
    with pytest.raises(MemoryError) as e_info:
        vxl.insight(grid, method=method, mem_limit=3000)

def test__geo_to_np_coordinate():
    raster = np.zeros((5, 5), dtype=np.uint8)
    raster[0, 0] = 42
    raster[4, 4] = 25
    raster[2, 0] = 7

    raster_truth = np.zeros_like(raster)
    raster_truth[-1, 0] = 42
    raster_truth[0, -1] = 25
    raster_truth[-1, 2] = 7

    assert (raster_truth == vxl._geo_to_np_coordinate(raster)).all(), 'Missmatch between 2D raters' 

    raster = np.zeros((5, 5, 3), dtype=np.uint8)
    raster[0, 0, 0] = 42
    raster[4, 4, 1] = 25
    raster[2, 0, 2] = 7

    raster_truth = np.zeros_like(raster)
    raster_truth[-1, 0, 0] = 42
    raster_truth[0, -1, 1] = 25
    raster_truth[-1, 2, 2] = 7

    assert (raster_truth == vxl._geo_to_np_coordinate(raster)).all(), 'Missmatch between 3D raters' 

@pytest.mark.parametrize('set_id, grid_id, axis, method', [
    (1, 1, 2, 'top'),
    (1, 1, 2, 'center'),
    (1, 1, 2, 'bottom'),
    (1, 1, 2, 'mean'),
    (1, 1, 2, 'max'),
    (1, 1, 2, 'min'),
    (1, 1, 2, 'median'),
    (1, 1, 0, 'top'),
    (1, 1, 0, 'center'),
    (1, 1, 0, 'bottom'),
    (1, 1, 1, 'top'),
    (1, 1, 1, 'center'),
    (1, 1, 1, 'bottom'),
])
def test_squash(datadir, set_id, grid_id, axis, method):
    vxld = data_vxl(datadir, set_id, grid_id, 'density' )
    truth = data_raster(datadir, set_id, grid_id, axis, method)
    res = vxl.squash(vxld, method, axis)

    assert res is not None, 'Tested function did not return anything :('
    assert res.shape == truth.shape, 'Missmatch between truth and tested shape'
    assert np.allclose(res, truth), 'Missmatch between truth and tested raster'
