#!/usr/bin/env python
# file rasterize-lidar.py
# author Florent Guiotte <florent.guiotte@irisa.fr>
# version 0.0
# date 18 févr. 2021
"""Create raster of LiDAR features.

Use npz point clouds from Idefix.
"""

import json
import argparse
from multiprocessing import Pool
from tqdm.auto import tqdm
from pathlib import Path

import idefix as ix

parser = argparse.ArgumentParser(description='Compute features rasters from .npz point cloud.',
                                 formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog="""
The config file can contain any parameters of the
idefix.helpers.rasterize function in a json file.

You can define 'global' parameters (for all the rasters) and raster
specific parameters in a list 'rasters'.

See the following `config.json` example file:

{
 "global": {
  "resolution": 5,
  "interpolation": "idw",
  "out_dir": "./results"
 },
 "rasters": [
  {
   "feature": "elevation",
   "bin_structure": "voxel",
   "bin_method": "mean",
   "squash_method": ["top", "bottom", "std"]
  },
  {
   "feature": "elevation",
   "bin_structure": "pixel",
   "bin_method": "mean"
  },
  {
   "bin_structure": "pixel",
   "bin_method": "density"
  },
  {
   "feature": "num_returns",
   "bin_structure": "pixel",
   "bin_method": "mode"
  }
 ]
}
""")

parser.add_argument('-c', '--config', type=str, help='json file to setup the rasterization processes', dest='c')
parser.add_argument('-n', '--nprocess', type=int, help='number of child processes to use', default=1, dest='n')
parser.add_argument('in_dir', type=str, help='the path to the point cloud directory')
parser.add_argument('out_dir', type=str, help='path to output raster results')

args = parser.parse_args()

def _map_rasterize(kwargs):
    return ix.helpers.rasterize(**kwargs)

def main():
    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(exist_ok=True)


    pc_files = list(in_dir.glob('*.npz'))


    config = {'global': {'out_dir': out_dir}}
    config_json = json.load(open(args.c)) if args.c else {}
    config['global'].update(config_json['global'] if 'global' in config_json else {})
    config['rasters'] = config_json['rasters'] if 'rasters' in config_json else {}

    globalc = config['global']
    rasters = config['rasters'] if 'rasters' in config else [{}]
    
    queue = []
    for pc_file in pc_files:
        for raster in rasters:
            job = globalc.copy()
            job.update(raster)
            job.update({'pc_file': pc_file})
            queue += [job]

    pool = Pool(processes=args.n)
    for _ in tqdm(pool.imap_unordered(_map_rasterize, queue), total=len(queue)):
        pass

if __name__ == '__main__':
    main()
