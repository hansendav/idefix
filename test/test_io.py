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
