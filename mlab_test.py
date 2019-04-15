#!/usr/bin/env python
# file mlab_test.py
# author Florent Guiotte <florent.guiotte@uhb.fr>
# version 0.0
# date 11 avril 2019
"""Abstract

doc.
"""

from idefix import vxl
import mayavi.mlab as mlab
import numpy as np

spatial = np.random.random((10000, 3))
feature = np.random.random(10000)

grid = vxl.get_grid(spatial, .1)
vg = vxl.bin(grid, spatial, feature, 'mean')

vxl.plot(vg)
mlab.show()
