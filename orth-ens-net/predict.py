import math
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import sys
from configparser import ConfigParser

import nibabel as nib
import numpy as np
from nibabel import load as load_nii
from utils import load_model, add_padding_z, add_padding_x, add_padding_y, \
    ensure_dir
from tensorflow.keras.backend import argmax
from keras.activations import sigmoid, softmax


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
            pd_z = (math.ceil(img_np.shape[0] / factor) * factor)
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


def load_and_predict_raw_image(subjects, model_n, fold, normalization_fn=None,
                               src_dir_path=None, im_types=None,
                               path_structure=None, has_logits=False,
                               dat_name=None):
    model_path = os.path.join(models_directory, fold, model_n)
    if not os.path.exists(model_path + '.json'):
        print(f"MODEL {model_path} not found. Skipping...")
        return
    model = load_model(model_path)

    for subject in subjects:
        subject = subject.strip('\n')
        save_folder = os.path.join(segmentation_directory,
                                   os.path.basename(subject), fold,
                                   model_n)
        print('subject: ', subject)
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

        imgs_np, pd_x, pd_y, pd_z = pad_if_necessary(imgs_np, 16)

        all_modalities_joined = np.stack(imgs_np).astype(np.float32)
        all_modalities_joined = np.moveaxis(all_modalities_joined, 0, -1)
        all_modalities_joined = np.expand_dims(all_modalities_joined, 0)

        logits = model.predict(all_modalities_joined, batch_size=1)

        activation = sigmoid if out_channels == 2 else softmax
        y_pred = activation(logits)

        if has_logits:
            if not type(y_pred) == list:
                print("Expected logits with output, but got only predictions."
                      " Setting save_logits to False.")
                has_logits = False
            else:
                y_pred, logits = y_pred

        y_pred = unpad_pred(y_pred, orig_shape, pd_x, pd_y, pd_z)
        if has_logits:
            logits = unpad_pred(logits, orig_shape, pd_x, pd_y, pd_z)

        y_pred = y_pred.astype(float)[0]  # (1, x, y, z, c) -> (x, y, z, c)
        if y_pred.shape[3] > 1:
            mask_img = argmax(y_pred)
        else:
            mask_img = np.array(y_pred >= 0.5).astype(float)[:, :, :, 0]
        if has_logits:
            logits_img = logits.astype(float)[0]

        ensure_dir(save_folder)
        nib.save(nib.Nifti1Image(y_pred, None, header_info),
                 f"{save_folder}/{dat_name}_prediction.nii.gz")

        nib.save(nib.Nifti1Image(mask_img, None, header_info),
                 f"{save_folder}/{dat_name}_mask.nii.gz")
        if has_logits:
            nib.save(nib.Nifti1Image(logits_img, None, header_info),
                     f"{save_folder}/{dat_name}_logits.nii.gz")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise AttributeError(f"When running 'python predict.py' you also"
                             f"must provide the path of the configuration file"
                             f" (.ini).")
    ini_file = sys.argv[1]
    parser = ConfigParser()
    parser.read(ini_file)

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

    images = parser['TEST'].get('images').split(',')
    imgs_paths = parser['TEST'].get('imgs_paths').replace('%workspace',
                                                          workspace_dir)

    out_channels = parser['TRAIN'].getint('output_channels')

    norm = parser['TEST'].get('normalization')
    normalization = NORMALIZATION[norm] if norm in NORMALIZATION else None

    logits = parser['TEST'].getboolean('save_logits')

    hold_out_txt = parser['DEFAULT'].get(
        'hold_out_data').replace('%workspace', workspace_dir)
    hold_out_file = open(hold_out_txt, "r")
    hold_out_images = hold_out_file.read().split(' ')

    for model_fold in model_folds:
        for i in range(n_models):
            model_name = 'model_{}'.format(i)
            load_and_predict_raw_image(hold_out_images, model_name, model_fold,
                                       normalization, origin_directory, images,
                                       imgs_paths, logits, name)
