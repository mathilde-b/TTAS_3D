#!/usr/bin/env python3.6
import argparse
from argparse import Namespace
import warnings
from sys import argv
from typing import Dict, Iterable
from pathlib import Path
from functools import partial

import numpy as np
from skimage.io import imread, imsave
from utils import mmap_, read_anyformat_image


def check(filename: str):
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)

        acc = read_anyformat_image(filename)
        acc = np.array(acc)
        #acc[acc==255]=1
        #acc=acc[:,:,0]
        #if len(np.unique(acc))>2:
        #print(np.unique(acc))
        print(np.max(acc))
        #print(acc.shape)
        #imsave(filename,acc)
        #print(np.min(acc),np.max(acc))
        


def main():

    folder = Path(argv[1])

    targets: Iterable[str] = map(str, folder.glob("*nii"))
    for target in targets:
        print(target)
        check(target)

if __name__ == "__main__":
    main()
