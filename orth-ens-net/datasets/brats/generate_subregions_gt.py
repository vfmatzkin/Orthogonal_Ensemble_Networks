# Not used anymore.

import os.path
import sys
from configparser import ConfigParser

import nibabel as nib
import numpy as np
from nibabel import load as load_nii


def or_img_labels(mask_img: np.array, combinations: dict, root_path: str):
    """ Combine labels using OR according to label combinations dictionary.

    This function only combines the images, but it won't save them.

    :param mask_img: 3D image of the prediction.
    :param combinations: Dictionary containing the name of the combination as
    key and the combined labels as values.
    :param root_path: Where to save the combined imgs.
    :return: dict containing mask output paths as keys and combined imgs as
    values.
    """
    combined_imgs = {}
    for name, labels in combinations.items():
        saved_path = os.path.join(root_path, f'{dataset}_{name}_gt.nii.gz')
        if len(labels) == 1:
            result_img = np.array(mask_img == labels[0], dtype=np.float)
        else:
            result_img = np.array(
                np.logical_or.reduce([mask_img == l for l in labels]),
                dtype=np.float)

        combined_imgs[saved_path] = result_img
    return combined_imgs


def generate_subregions_gt():
    for subject in hold_out_images:
        subject = subject.strip('\n')
        subj_folder = os.path.join(origin_directory, subject)
        seg_path = os.path.join(subj_folder, subject + '_seg.nii.gz')

        print('subject: ', subject)

        out_paths = []
        for name in comb_labels.keys():
            out_path = os.path.join(subj_folder, f'{dataset}_{name}_gt.nii.gz')
            out_paths.append(out_path)

        exist = all([os.path.exists(pth) for pth in out_paths])

        if exist and not overwrite:
            print(f"Skipping patient because overwrite is disabled and the "
                  f"out imgs already exist. Run with --overwrite to force. "
                  f"Found:\n"
                  f"{os.linesep.join(out_paths)}\n")
            continue

        img_nii = load_nii(seg_path)
        img_np = img_nii.get_fdata()
        header_info = img_nii.header

        cmb_mask = or_img_labels(img_np, comb_labels, subj_folder)
        for path, img in cmb_mask.items():
            nib.save(nib.Nifti1Image(img, None, header_info), path)
            out_paths.append(path)


if __name__ == '__main__':
    if len(sys.argv) < 2:
        raise AttributeError(
            f"When running 'python generate_subregions_gt.py' "
            f"you also must provide the path of the "
            f"configuration file (.ini).")
    ini_file = sys.argv[1]
    parser = ConfigParser()
    parser.read(ini_file)

    overwrite = '--overwrite' in sys.argv

    workspace_dir = parser['DEFAULT'].get('workspace_dir')
    origin_directory = parser['DEFAULT'].get('image_source_dir').replace(
        '%workspace', workspace_dir)

    out_channels = parser["TRAIN"].getint("output_channels")
    if 'labels' in parser['TRAIN']:  # labels='1,2,4' --> labels = [1, 2, 4]
        lab = parser['TRAIN']['labels']
        labels = list(map(int, lab.split(',')))
    else:
        labels = list(range(out_channels))  # out_ch=3 -> labels = [0,1,2]

    dataset = parser['ENSEMBLE'].get('dataset')

    combine_labels = None if 'combine_labels' not in parser['TEST'] else \
        parser['TEST']['combine_labels']

    if combine_labels is not None and 'labels' in parser['TRAIN']:
        print("WARNING: Taking combine_labels values as the labels in the "
              f"changed labels file ({labels} and not "
              f"{list(range(out_channels))}.).")

    comb_labels = {e.split(':')[0]: e.split(':')[1].split(',') for
                   e in combine_labels.split(';')}
    comb_labels = {k: [int(l) for l in lbl] for k, lbl in
                   comb_labels.items()}
    if len(comb_labels) == 0:  # Found nothing
        raise AttributeError(f"combine_labels param is "
                             f"{combine_labels} but is expected a str"
                             f" like 'ET:1;TC:1,4;WT:1,2,4'. If you "
                             f"don't want to combine unset this param")

    images = parser['TEST'].get('images').split(',')
    imgs_paths = parser['TEST'].get('imgs_paths').replace('%workspace',
                                                          workspace_dir)

    hold_out_txt = parser['DEFAULT'].get(
        'hold_out_data').replace('%workspace', workspace_dir)
    hold_out_file = open(hold_out_txt, "r")
    hold_out_images = hold_out_file.read().split(' ')

    generate_subregions_gt()
