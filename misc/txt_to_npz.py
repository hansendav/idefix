#!/usr/bin/env python
# file txt_to_npz.py
# author Florent Guiotte <florent.guiotte@uhb.fr>
# version 0.0
# date 24 mai 2019
"""Convert point clouds from text files to Idefix file format.

doc.
"""

import numpy as np
import idefix.io as io
from pathlib import Path
import argparse
from tqdm import tqdm

def txt_to_npy(fname, header, delimiter=None, dtype=None, compression=False):
    oname = fname.stem + '.npz'
    pc = io.load_txt(fname, header, delimiter, dtype)
    io.dump_pc(oname, pc, compression)

def main():
    parser = argparse.ArgumentParser(description='Convert point clouds from text files to Idefix file format.')
    parser.add_argument('file', type=str, help='file or dir to convert')
    parser.add_argument('header', type=str, help='field names of the data')
    parser.add_argument('--dtype', '-t', type=str, help='field data types')
    parser.add_argument('--delimiter', '-d', type=str, default=',', help='field data delimiter')
    parser.add_argument('--compress', '-c', action='store_true', default=False, help='enable data compression')

    args = parser.parse_args()
    header = args.header.split()
    dtype = [np.dtype(x) for x in args.dtype.split()] if args.dtype else None
    delimiter = args.delimiter
    compress = args.compress
    wd = Path(args.file)

    if wd.is_dir():
        files = wd.glob('*.txt')
    else:
        files = (wd,)

    pbar = tqdm(list(files))
    for f in pbar:
        pbar.write('Processing {}...'.format(f))
        txt_to_npy(f, header, delimiter, dtype, compress)

if __name__ == '__main__':
    main()
