#!/usr/bin/env python
# file conftest.py
# author Florent Guiotte <florent.guiotte@uhb.fr>
# version 0.0
# date 04 mars 2019
"""Configuration script for PyTest.

Define in this script:
    - Fixtures shared among tests
    - Helpers functions for tests
    - External plugins
    - Hooks
"""

from distutils import dir_util
from pytest import fixture
import os

@fixture
def datadir(tmpdir, request):
    '''
    Fixture responsible for searching a folder with the same name of test
    module and, if available, moving all contents to a temporary directory so
    tests can use them freely.

    from: https://stackoverflow.com/a/29631801
    '''
    filename = request.module.__file__
    test_dir, _ = os.path.splitext(filename)

    if os.path.isdir(test_dir):
        dir_util.copy_tree(test_dir, str(tmpdir))

    return tmpdir

