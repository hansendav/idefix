#!/usr/bin/env python
# file test_io.py
# author Florent Guiotte <florent.guiotte@uhb.fr>
# version 0.0
# date 04 mars 2019
"""Idefix IO tests set.

doc.
"""

import pytest
import numpy as np
from idefix import io

@pytest.mark.parametrize('fname, exp_point_count, exp_field_count', [
    # TODO: test different LAS version
    # TODO: test LAS without field
    ('test.las', 58629, 3, ),
    #('test.laz', 58629, 3, ),
])
def test_load_las(datadir, fname, exp_point_count, exp_field_count):
    fname = datadir.join(fname)
    
    # Raise "No such file"
    with pytest.raises(IOError) as e_info:
        io.load_las('not_as_file.las')

    # Open file without exception
    try:
        result = io.load_las(fname)
    except IOError:
        pytest.fail('Opening legit file without exception')

    assert result.size == exp_point_count, "Return correct point count"

    assert result['spatial'].shape[-1] == 3, "Return ndarray with spatial field"

    assert (result['spatial'] == result.spatial).all(), "Quick access with records array"

    assert len(result['feature'].dtype) == exp_field_count, "Return ndarray with attribute fields"

    assert result.spatial.dtype == np.float, "Dtype of spatial is np.float"

@pytest.mark.parametrize('fname, head, separator, exp_point_count, exp_field_count, dtype', [
    # TODO: test different LAS version
    # TODO: test LAS without field
    ('test.txt', ['x', 'y', 'z', 'class', 'intensity'], ',', 58629, 2, None),
    ('test_b.txt', ['x', 'y', 'z', 'class', 'intensity'], ' ', 58629, 2, None),
    ('test.txt', ['x', 'y', 'z', 'class', 'intensity'], ',', 58629, 2, [np.float, np.float, np.float, np.uint8, np.uint8]),
    #('test.laz', 58629, 3, ),
])
def test_load_txt(datadir, fname, head, separator, exp_point_count, exp_field_count, dtype):
    fname = datadir.join(fname)
    
    # Raise "No such file"
    with pytest.raises(IOError) as e_info:
        io.load_txt('not_as_file.txt', head)

    # Raise "Header and file mismatch"
    with pytest.raises(IOError) as e_info:
        io.load_txt(fname, header=['x'])


    # Open file without exception
    try:
        result = io.load_txt(fname, head, separator)
    except IOError:
        pytest.fail('Opening legit file without exception')

    try:
        result = io.load_txt(fname, tuple(head), separator, dtype)
    except Exception:
        pytest.fail('Opening legit file with legit header')

    assert result.size == exp_point_count, "Return correct point count"

    assert result['spatial'].shape[-1] == 3, "Return ndarray with spatial field"

    assert (result['spatial'] == result.spatial).all(), "Quick access with records array"

    assert len(result['feature'].dtype) == exp_field_count, "Return ndarray with attribute fields"

    assert result.spatial.dtype == np.float, "Dtype of spatial is np.float"

    if dtype is not None:
        for feature_name, feature_dtype in zip(head[3:], dtype[3:]):
            assert result.feature[feature_name].dtype == feature_dtype, "Missmatch between specified dtype and returned feature dtype"

@pytest.mark.parametrize('fname, exp_point_count, exp_field_count', [
    ('test.npz', 58629, 2, ),
    ('test_compressed.npz', 58629, 2,),
    ('test_multi.npz', (100, 200), 2,),
    ('test_multi_compressed.npz', (100, 200), 2,),
])
def test_load_pc(datadir, fname, exp_point_count, exp_field_count):
    fname = datadir.join(fname)
    
    # Raise "No such file"
    with pytest.raises(IOError) as e_info:
        io.load_pc('not_as_file.npz')

    # Open file without exception
    try:
        result = io.load_pc(fname)
    except IOError:
        pytest.fail('Opening legit file without exception')

    if isinstance(exp_point_count, tuple):
        assert isinstance(result, tuple), "Multi point cloud file should return tuple of point cloud"
        result = result[0]
        exp_point_count = exp_point_count[0]

    assert result.size == exp_point_count, "Return correct point count"

    assert result['spatial'].shape[-1] == 3, "Return ndarray with spatial field"

    assert result.spatial.shape[-1] == 3, "Returned array is not a recarray"

    assert (result['spatial'] == result.spatial).all(), "Quick access with records array"

    assert len(result['feature'].dtype) == exp_field_count, "Return ndarray with attribute fields"

    assert result.spatial.dtype == np.float, "Dtype of spatial is np.float"

@pytest.mark.parametrize('fname, compress', [
    ('test.npz', False,),
    ('test.npz', True,),
    ('test_multi.npz', False,),
    ('test_multi.npz', True,),
])
def test_dump_pc(datadir, fname, compress):
    in_fname = datadir.join(fname)
    pc = io.load_pc(in_fname)

    out_fname = datadir / 'PYTEST_test.npz'

    try: 
        io.dump_pc(out_fname, pc, compress)
    except IOError:
        pytest.fail('Dump file without exception')

    assert out_fname.exists(), 'The dump file was not created'

    in_out_pc = io.load_pc(out_fname)

    assert len(in_out_pc) == len(pc), 'Missmatch of dumped point cloud'

    if isinstance(pc, tuple):
        assert in_out_pc[0].spatial.shape == pc[0].spatial.shape, 'Missmatch of dumped point cloud'
        assert in_out_pc[0].spatial.dtype == pc[0].spatial.dtype, 'Missmatch of dumped point cloud'
        assert in_out_pc[0].feature.dtype == pc[0].feature.dtype, 'Missmatch of dumped point cloud'
    else:
        assert in_out_pc.spatial.shape == pc.spatial.shape, 'Missmatch of dumped point cloud'
        assert in_out_pc.spatial.dtype == pc.spatial.dtype, 'Missmatch of dumped point cloud'
        assert in_out_pc.feature.dtype == pc.feature.dtype, 'Missmatch of dumped point cloud'
