""" check_datasets.py
This script checks that the npz files in a dataset are correct.

Usage: python check_datasets.py [DATASET_FOLDER]
"""

import os
import numpy as np
import sys


def check_dataset(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"The provided folder does not exist ({path})"
                                ".")
    if not os.path.isdir(path):
        raise AttributeError("The input path does not correspond to a folder"
                             f"({path}).")

    for file in os.listdir(path):
        full_path = os.path.join(path, file)
        if full_path.endswith('.npz'):
            print(f'Opening {full_path}.', end='. ')
            archive = np.load(full_path)
            archive.close()
            print('Closed sucessfully.')


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise AttributeError("You must enter the dataset folder path as "
                             "argument.")

    check_dataset(sys.argv[1])
