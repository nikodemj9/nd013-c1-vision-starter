import argparse
import glob
import os
import random

import numpy as np

from utils import get_module_logger


def split(source, destination):
    """
    Create three splits from the processed records. The files should be moved to new folders in the
    same directory. This folder should be named train, val and test.

    args:
        - source [str]: source data directory, contains the processed tf records
        - destination [str]: destination data directory, contains 3 sub folders: train / val / test
    """
    # Create folders, catch exception if folder is already created
    try:
        os.mkdir(destination+"/train")
    except OSError as error:
        print(error)
    try:
        os.mkdir(destination+"/val")
    except OSError as error:
        print(error)
    try:
        os.mkdir(destination+"/test")
    except OSError as error:
        print(error)

    # Get list of all files and shuffle
    files = os.listdir(source)
    random.shuffle(files) 

    # Split into 70-15-15
    files_100 = len(files)
    files_70 = int(0.7 * len(files))
    files_85 = int(0.85 * len(files))

    # Move training set (0-70%)
    for i in range(files_70):
        original_file = source+'/'+files[i]
        moved_file = destination+"/train/"+files[i]
        os.rename(original_file, moved_file)

    # Move evaluation set (70-85%)
    for i in range(files_70, files_85):
        original_file = source+'/'+files[i]
        moved_file = destination+"/val/"+files[i]
        os.rename(original_file, moved_file)

    # Move test set (85-100%)
    for i in range(files_85, files_100):
        original_file = source+'/'+files[i]
        moved_file = destination+"/test/"+files[i]
        os.rename(original_file, moved_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Split data into training / validation / testing')
    parser.add_argument('--source', required=True,
                        help='source data directory')
    parser.add_argument('--destination', required=True,
                        help='destination data directory')
    args = parser.parse_args()

    logger = get_module_logger(__name__)
    logger.info('Creating splits...')
    split(args.source, args.destination)