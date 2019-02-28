#!/usr/bin/python
# -*- coding: utf-8 -*-
# \file %filename%.py
# \brief TODO
# \author Florent Guiotte <florent.guiotte@gmail.com>
# \version 0.1
# \date 27 févr. 2019
#
# TODO details

from idefix import utils

def test_first():
    assert utils.first(1) == -1
    assert utils.first(-4) == 4
