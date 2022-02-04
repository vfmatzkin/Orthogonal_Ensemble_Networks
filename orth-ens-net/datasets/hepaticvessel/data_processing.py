""" Sort medicaldecathlon task8 dataset (HepaticVessel) files.

This script grabs the .tar file (or the extracted folder) and prepares the
files to the patch generation process.

Usage: python hepaticvessel.py [.TAR FILE|EXTRACTED FOLDER]

Go to http://medicaldecathlon.com/ for downloading the data
(Task08_HepaticVessel.tar).
"""

import os
import sys
import tarfile
import numpy as np
from ..build_dataset import build_dataset


def sort_folders(path, suffs):
    """Sort content by patient.

    Given a folder with many images of different patients and types of images,
    sort them in folders for each patient. Note that the images for each
    patient must differ in the given suffixes.

    Example: given the following folder:
    - images
        - sub001_ct.nii.gz
        - sub001_seg.nii.gz
        - sub002_ct.nii.gz
        - sub002_seg.nii.gz

    running sort_folders('~/folder', ['_ct.nii.gz', '_seg.nii.gz'])  will
    produce:
    - images
        - sub001
            - sub001_ct.nii.gz
            - sub001_seg.nii.gz
        - sub002
            - sub002_ct.nii.gz
            - sub002_seg.nii.gz

    :param path: input path
    :param suffs: list of image suffixes
    :return:
    """
    files = os.listdir(path)
    for file in files:
        contains = [file.endswith(s) for s in suffs]
        if not any(contains):
            supp_exts = ", ".join(suffs)
            print(f"The file {file} does not contain any of the provided "
                  f"suffixes ({supp_exts}), so it will not be moved.")
            continue
        suff = suffs[np.where(contains)[0][0]]
        folder_name = file[0:-len(suff)]
        new_folder_path = os.path.join(path, folder_name)
        os.makedirs(new_folder_path, exist_ok=True)
        full_file_path = os.path.join(path, file)
        new_file_path = os.path.join(new_folder_path, file)
        print(f"Moving {full_file_path} to {new_file_path}")
        os.rename(full_file_path, new_file_path)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise AttributeError("Enter the path of the .tar file or the excracted"
                             " folder. ")
    input_path = sys.argv[1]
    if input_path.endswith('.tar'):
        print(f"Extracting {input_path}...")
        tar = tarfile.open(input_path)
        extr_folder = input_path[:-4]
        tar.extractall(path=extr_folder)
    else:
        extr_folder = input_path

    folders = [os.path.join(extr_folder, s) for s in ['imagesTr', 'labelsTr']]
    for folder in folders:
        if not os.path.exists(folder):
            raise FileNotFoundError(f"The folder {folder} does not exist. Make"
                                    " sure you entered a folder that contains "
                                    "the imagesTr and labelsTr folders.")
        for image in os.listdir(folder):
            old_path = os.path.join(folder, image)
            new_suffix = f'_{"ct" if "imagesTr" in folder else "seg"}.nii.gz'
            new_path = old_path.replace('.nii.gz', new_suffix)
            if 'labelsTr' in new_path:
                new_path = new_path.replace('labelsTr', 'imagesTr')
            os.rename(old_path, new_path)

    sort_folders(folders[0], ['_ct.nii.gz', '_seg.nii.gz'])
    print(f'Sorted images saved in {folders[0]}')

    origin_directory = folders[0]
    patches_directory = os.path.join(origin_directory, 'patches')
    build_dataset(origin_directory, patches_directory, 'hepaticvessel')
