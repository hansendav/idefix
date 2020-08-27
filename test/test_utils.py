#!/usr/bin/env python
# file test_utils.py
# author Florent Guiotte <florent.guiotte@uhb.fr>
# version 0.0
# date 28 févr. 2019
"""General utils functions unitary tests.

"""

import numpy as np
import pytest
from idefix import utils

@pytest.mark.parametrize("first_input,first_expected", [
    (1, -1),
    (-4, 4),
])
def test_first(first_input, first_expected):
    assert utils.first(first_input) == first_expected

@pytest.fixture
def fix_data():
    np.random.seed(0)
    return np.random.random((10,3))

def test_bbox(fix_data):
    res = np.array([fix_data.min(axis=0), fix_data.max(axis=0)])
    assert (utils.bbox(fix_data) == res).all()

def test_read(datadir):
    with open(datadir.join('first.txt')) as f:
        assert f.read() == 'hullo\n'
