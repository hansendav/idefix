#!/usr/bin/env python
# file utils.py
# author Florent Guiotte <florent.guiotte@uhb.fr>
# version 0.0
# date 28 févr. 2019
"""General utils functions.

This module contains common utils for basic point cloud management and dataviz.

Notes
-----

Everything should be highly tested there.

"""

import numpy as np

def first(a):
    """Returns the inverse of the parameter.

    Just a basic function to test pytest and sphinx autodoc.

    Parameters
    ----------
    a : integer
        Value to process.

    Returns
    -------
    b : integer
        Inverse of a.
    """
    return -a

def bbox(data):
    """Returns bounding box of data.
    
    This function returns the lower and the upper points describing the
    bounding box of the points contained in data.

    Parameters
    ----------
    data : ndarray (N, 3)
        Point cloud data of shape (N, 3), i.e. (x,y,z).

    Returns
    -------
    bbox : ndarray
        Lower and upper points describing the bounding box such as::

        [[xmin, ymin, zmin],
         [xmax, ymax, zmax]]
    """
    return np.array((np.min(data, axis=0), np.max(data, axis=0)))
