#!/usr/bin/env python
# file utils.py
# author Florent Guiotte <florent.guiotte@uhb.fr>
# version 0.0
# date 28 févr. 2019
"""General utils functions.

This module contains common utils for basic point cloud management and dataviz.

Notes
-----

Everything is well tested there.

"""

import numpy as np
import logging

log = logging.getLogger(__name__)

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
    log.info('first called.')
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

    See Also
    --------
    fit_bbox : Return a bounding box fit on fixed coordinates.
    """
    return np.array((np.min(data, axis=0), np.max(data, axis=0)))

def fit_bbox(data, decimals=0):
    """Return a bounding box fit on fixed coordinates.

    - Round $x$ and $y$ coordinates to match most orthoimagery tiles.
    - Ceil and floor $z$ coordinates to include all the point on the vertical axis.

    Parameters
    ----------
    data : ndarray (N, 3)
        Bbox or point cloud data of shape (N, 3), i.e. (x,y,z).
    decimals : int
        The precision for the rounding, ceiling and flooring operations.

    Returns
    -------
    bbox : ndarray
        Lower and upper points describing the bounding box such as::

        [[xmin, ymin, zmin],
         [xmax, ymax, zmax]]

    See Also
    --------
    bbox : Returns a raw bounding box on the data.
    """
    bbox = bbox(data)

    nbbox = np.round(bbox, decimals)
    nbbox[0,2] = np.floor(bbox * 10 ** decimals)[0,2] / 10 ** decimals
    nbbox[1,2] = np.ceil(bbox * 10 ** decimals)[1,2] / 10 ** decimals

    return nbbox
