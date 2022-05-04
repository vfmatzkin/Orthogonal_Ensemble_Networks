import math
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import sys
from configparser import ConfigParser

import nibabel as nib
import numpy as np
from nibabel import load as load_nii
from utils import load_model, add_padding_z, add_padding_x, add_padding_y, \
    ensure_dir, or_img_labels, comb_probs
import tensorflow.keras.backend as K
from keras.activations import sigmoid, softmax
import torchio as tio
import torch


def z_scores_normalization(img):
    img = (img - np.mean(img)) / np.std(img)
    return img


NORMALIZATION = {None: None,
                 'z_scores_normalization': z_scores_normalization}


def pad_if_necessary(imgs_np, factor):
    pd_x, pd_y, pd_z = None, None, None
    for i, img_np in enumerate(imgs_np):
        if (img_np.shape[0] % factor) != 0:
            pd_x = (math.ceil(img_np.shape[0] / factor) * factor)
            imgs_np[i] = add_padding_x(imgs_np[i], pd_x)
        if (img_np.shape[1] % factor) != 0:
            pd_y = (math.ceil(img_np.shape[1] / factor) * factor)
            imgs_np[i] = add_padding_y(imgs_np[i], pd_y)
        if (img_np.shape[2] % factor) != 0:
            pd_z = (math.ceil(img_np.shape[2] / factor) * factor)
            imgs_np[i] = add_padding_z(imgs_np[i], pd_z)
    return imgs_np, pd_x, pd_y, pd_z


def unpad_pred(y_pred, orig_shape, pd_x, pd_y, pd_z):
    if pd_x != None:
        y_pred = y_pred[:, np.floor_divide(
            (pd_x - orig_shape[0]), 2):-math.ceil(
            (pd_x - orig_shape[0]) / 2), :, :, :]
    if pd_y != None:
        y_pred = y_pred[:, :, np.floor_divide(
            (pd_y - orig_shape[1]), 2):-math.ceil(
            (pd_y - orig_shape[1]) / 2), :, :]
    if pd_z != None:
        y_pred = y_pred[:, :, :, np.floor_divide(
            (pd_z - orig_shape[2]), 2):-math.ceil(
            (pd_z - orig_shape[2]) / 2), :]
    return y_pred


def predict_tio(model, all_modalities_joined, patch_size=128, patch_overlap=4):
    tio_image = tio.ScalarImage(tensor=np.moveaxis(all_modalities_joined[0], -1, 0))
    subject = tio.Subject(image=tio_image)

    sampler = tio.inference.GridSampler(subject, patch_size, patch_overlap, padding_mode=0)
    aggregator = tio.inference.GridAggregator(sampler, overlap_mode='average')
    for i, patch in enumerate(sampler):
        input_tensor = patch['image'][tio.DATA]
        locations = patch[tio.LOCATION].unsqueeze(0)  # Add batch dimension
        expanded_patch = np.expand_dims(np.moveaxis(input_tensor.numpy(), 0, 3), axis=0)
        logits_keras = model.predict(expanded_patch, batch_size=1)
        logits_pt = torch.tensor(logits_keras).permute(0, 4, 1, 2, 3)
        aggregator.add_batch(logits_pt, locations)
    output_tensor = aggregator.get_output_tensor()
    return np.expand_dims(output_tensor.permute(1, 2, 3, 0).numpy(), 0)


def load_and_predict_raw_image(subjects, model_n, fold, normalization_fn=None,
                               src_dir_path=None, im_types=None,
                               path_structure=None, save_logits=False,
                               dat_name=None, overwrite=False,
                               patch_based=False):
    model_path = os.path.join(models_directory, fold, model_n)
    if not os.path.exists(model_path + '.json'):
        print(f"MODEL {model_path} not found. Skipping...")
        return
    model = load_model(model_path)

    if patch_based:
        print("Using patch based inference.")

    for subject in subjects:
        subject = subject.strip('\n')
        print('subject: ', subject)
        save_folder = os.path.join(segmentation_directory,
                                   os.path.basename(subject), fold,
                                   model_n)

        # Images to save - Check if they exist
        out_pred_path = f"{save_folder}/{dat_name}_prediction.nii.gz"
        out_mask_path = f"{save_folder}/{dat_name}_mask.nii.gz"
        out_mask_lbl_path = f"{save_folder}/{dat_name}_mask_lbl.nii.gz"
        out_logits_path = f"{save_folder}/{dat_name}_logits.nii.gz"
        out_paths = [out_pred_path, out_mask_path] if not save_logits \
            else [out_pred_path, out_mask_path, out_logits_path]
        exist = all([os.path.exists(pth) for pth in out_paths])
        if exist and not overwrite:
            print(f"Skipping patient because overwrite is disabled and the "
                  f"out imgs already exist. Run with --overwrite to force."
                  f"Found:\n"
                  f"{os.linesep.join(out_paths)}\n")
            continue

        print('output_folder: ', save_folder)

        im_paths = {}  # Paths of the images to load
        if path_structure:
            for image in im_types:
                im_fldr = path_structure.replace('%origdir', src_dir_path)
                im_fldr = im_fldr.replace('%subject', subject)
                im_fldr = im_fldr.replace('%image', image)
                im_paths[image] = im_fldr

        else:
            for image in im_types:
                im_paths[image] = os.path.join(src_dir_path, image)

        imgs_np = []
        for im_name, im_path in im_paths.items():
            img_nii = load_nii(im_path)  # Not used anymore later
            header_info = img_nii.header  # Same for all the files? Used once
            imgs_np.append(np.array(img_nii.get_fdata()))
        orig_shape = imgs_np[0].shape  # Used once

        if normalization_fn:
            imgs_np = [normalization_fn(im) for im in imgs_np]

        # If patch based, the patch size cannot be bigger than any image
        # dimension, so the image size should be padded by the LCM of 16 & PS.
        multiple_of = 16 if not patch_based else np.lcm(16, patch_size)
        imgs_np, pd_x, pd_y, pd_z = pad_if_necessary(imgs_np, multiple_of)

        all_modalities_joined = np.stack(imgs_np).astype(np.float32)
        all_modalities_joined = np.moveaxis(all_modalities_joined, 0, -1)
        all_modalities_joined = np.expand_dims(all_modalities_joined, 0)

        if not patch_based:  # Use whole image
            logits = model.predict(all_modalities_joined, batch_size=1)
        else:  # Use torchio patch based inference
            logits = predict_tio(model, all_modalities_joined, patch_size)

        activation = sigmoid if out_channels == 2 else softmax
        y_pred = activation(K.constant(logits)).numpy()

        y_pred = unpad_pred(y_pred, orig_shape, pd_x, pd_y, pd_z)
        if save_logits:
            logits = unpad_pred(logits, orig_shape, pd_x, pd_y, pd_z)

        y_pred = y_pred.astype(float)[0]  # (1, x, y, z, c) -> (x, y, z, c)
        if y_pred.shape[3] > 1:
            mask_img = K.argmax(y_pred)
        else:
            mask_img = np.array(y_pred >= 0.5).astype(float)[:, :, :, 0]
        if save_logits:
            logits_img = logits.astype(float)[0]

        cond_list = [mask_img == i for i in range(out_channels)]
        y_pred_lbls = np.select(cond_list, labels, mask_img)  # Corrected lbls

        ensure_dir(save_folder)
        nib.save(nib.Nifti1Image(y_pred, None, header_info), out_pred_path)
        nib.save(nib.Nifti1Image(mask_img, None, header_info), out_mask_path)
        if save_logits:
            nib.save(nib.Nifti1Image(logits_img, None, header_info),
                     out_logits_path)

        changed_labels = np.any(y_pred != y_pred_lbls)  # Any label has changed
        if changed_labels:  # If labels are different from the orig img
            nib.save(nib.Nifti1Image(y_pred_lbls, None, header_info),
                     out_mask_lbl_path)
            out_paths = out_paths + [out_mask_lbl_path]

        if combine_labels:  # Save combinations of labels
            comb_labels = {e.split(':')[0]: e.split(':')[1].split(',') for
                           e in combine_labels.split(';')}
            comb_labels = {k: [int(l) for l in lbl] for k, lbl in
                           comb_labels.items()}
            if len(comb_labels) == 0:  # Found nothing
                raise AttributeError(f"combine_labels param is "
                                     f"{combine_labels} but is expected a str"
                                     f" like 'ET:1;TC:1,4;WT:1,2,4'. If you "
                                     f"don't want to combine unset this param")
            save_path = f"{save_folder}/{dat_name}"

            # Combined probabilities and masks.
            cmb_probs = comb_probs(y_pred, comb_labels, save_path, labels)
            cmb_mask = or_img_labels(y_pred_lbls, comb_labels, save_path)
            cmb_probs.update(cmb_mask)  # Merge dicts
            for path, img in cmb_probs.items():
                nib.save(nib.Nifti1Image(img, None, header_info), path)
                out_paths.append(path)

        print(f"Saved files: {os.linesep + ' '} "
              f"{(os.linesep + '  ').join(out_paths)}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise AttributeError(f"When running 'python predict.py' you also"
                             f"must provide the path of the configuration file"
                             f" (.ini).")
    ini_file = sys.argv[1]
    parser = ConfigParser()
    parser.read(ini_file)

    overwrite = '--overwrite' in sys.argv

    workspace_dir = parser['DEFAULT'].get('workspace_dir')
    origin_directory = parser['DEFAULT'].get('image_source_dir').replace(
        '%workspace', workspace_dir)
    segmentation_directory = parser['DEFAULT'].get(
        'segmentation_directory').replace('%workspace', workspace_dir)
    models_directory = parser['DEFAULT'].get('models_directory').replace(
        '%workspace', workspace_dir)
    model_folds = parser['ENSEMBLE'].get('pretrained_models_folds').split(",")
    n_models = parser['ENSEMBLE'].getint('n_models')
    name = parser['ENSEMBLE'].get('dataset')

    out_channels = parser["TRAIN"].getint("output_channels")
    if 'labels' in parser['TRAIN']:  # labels='1,2,4' --> labels = [1, 2, 4]
        lab = parser['TRAIN']['labels']
        labels = list(map(int, lab.split(',')))
    else:
        labels = list(range(out_channels))  # out_ch=3 -> labels = [0,1,2]

    images = parser['TEST'].get('images').split(',')
    imgs_paths = parser['TEST'].get('imgs_paths').replace('%workspace',
                                                          workspace_dir)

    norm = parser['TEST'].get('normalization')
    normalization = NORMALIZATION[norm] if norm in NORMALIZATION else None

    logits = parser['TEST'].getboolean('save_logits')
    combine_labels = None if 'combine_labels' not in parser['TEST'] else \
        parser['TEST']['combine_labels']

    if combine_labels is not None and 'labels' in parser['TRAIN']:
        print("WARNING: Taking combine_labels values as the labels in the "
              f"changed labels file ({labels} and not "
              f"{list(range(out_channels))}.).")

    patch_based = False if 'patch_based' not in parser['TEST'] else \
        parser['TEST'].getboolean('patch_based')
    patch_size = parser['TEST'].getint('patch_size') \
        if 'patch_size' in parser['TEST'] else 128

    hold_out_txt = parser['DEFAULT'].get(
        'hold_out_data').replace('%workspace', workspace_dir)
    hold_out_file = open(hold_out_txt, "r")
    hold_out_images = hold_out_file.read().split(' ')

    for model_fold in model_folds:
        for i in range(n_models):
            model_name = 'model_{}'.format(i)
            load_and_predict_raw_image(hold_out_images, model_name, model_fold,
                                       normalization, origin_directory, images,
                                       imgs_paths, logits, name, overwrite,
                                       patch_based)
