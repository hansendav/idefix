#!/usr/bin/python
# -*- coding: utf-8 -*-
# \file setup.py
# \brief TODO
# \author Florent Guiotte <florent.guiotte@gmail.com>
# \version 0.1
# \date 11 sept. 2018
#
# TODO details

from distutils.core import setup

setup(name='idefix',
      version='1.4',
      description='Utils and processing pipelines for LiDAR point clouds',
      author='Florent Guiotte',
      author_email='florent.guiotte@uhb.fr',
      url='https://git.guiotte.fr/Florent/Idefix',
      packages=['idefix', 'idefix.tools'],
      entry_points = {'console_scripts':['txt2npz = idefix.tools.txt_to_npz:main',]},
      )
