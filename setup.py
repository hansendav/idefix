#!/usr/bin/python
# -*- coding: utf-8 -*-
# \file setup.py
# \brief TODO
# \author Florent Guiotte <florent.guiotte@gmail.com>
# \version 0.1
# \date 11 sept. 2018
#
# TODO details

import setuptools

with open('README.md', 'r') as fh:
    long_description = fh.read()

setuptools.setup(
    name='idefix',
    version='0.2.0',
    description='Utils and processing pipelines for LiDAR point clouds',
    author='Florent Guiotte',
    author_email='florent.guiotte@uhb.fr',
    url='https://git.guiotte.fr/Florent/Idefix',
    long_description=long_description,
    long_description_content_type='text/markdown',
    packages=['idefix', 'idefix.tools'],
    entry_points = {'console_scripts':[
        'txt2npz=idefix.tools.txt_to_npz:main',
        'rasterize=idefix.tools.rasterize:main',
    ]},
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
    install_requires=[
        'numpy',
        'sap',
        'tqdm',
        'matplotlib',
        'pathlib',
        'rasterio',
        'laspy==1.7.0',
        'humanize',
        #'mayavi', Optional, for vxl.plot()
    ],
      )
