import json
import math
import os
from typing import Union

import nibabel as nib
import numpy as np
from tensorflow.keras.models import model_from_json


def save_model(model_input, model_name_input='model'):
    model_json = model_input.to_json()
    model_json = json.loads(model_json)
    model_json['class_name'] = 'Model'  # this attr sometimes isnt properly set
    model_json = json.dumps(model_json)
    with open(model_name_input + ".json", "w") as json_file:
        json_file.write(model_json)
    model_input.save_weights(model_name_input + ".h5")
    print("Saved " + model_name_input)


def load_model(model_name_input='model'):
    json_file = open(model_name_input + '.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights(model_name_input + ".h5")
    print("Loaded model from disk")
    return loaded_model


def ensure_dir(directory):
    os.makedirs(directory, exist_ok=True)


def add_padding_z(img, depth_with_padding):
    pad_z = depth_with_padding
    pad_value = img[0][0][0]

    image_padded = np.empty((img.shape[0], img.shape[1], pad_z))
    image_padded.fill(pad_value)
    image_padded[:, :,
    np.floor_divide((depth_with_padding - img.shape[2]), 2):-math.ceil(
        (depth_with_padding - img.shape[2]) / 2)] = img
    return image_padded


def add_padding_x(img, depth_with_padding):
    pad_x = depth_with_padding

    pad_value = img[0][0][0]

    image_padded = np.empty((pad_x, img.shape[1], img.shape[2]))
    image_padded.fill(pad_value)
    image_padded[
    np.floor_divide((depth_with_padding - img.shape[0]), 2):-math.ceil(
        (depth_with_padding - img.shape[0]) / 2), :, :] = img
    return image_padded


def add_padding_y(img, depth_with_padding):
    pad_y = depth_with_padding

    pad_value = img[0][0][0]

    image_padded = np.empty((img.shape[0], pad_y, img.shape[2]))
    image_padded.fill(pad_value)
    image_padded[:,
    np.floor_divide((depth_with_padding - img.shape[1]), 2):-math.ceil(
        (depth_with_padding - img.shape[1]) / 2), :] = img
    return image_padded


def multi_class_prediction(prediction):
    prediction_images = []
    for i in range(prediction.shape[4]):
        prediction_images.append(
            nib.Nifti1Image(prediction[0, :, :, :, i], None))


def one_hot_labels(data, n_labels, labels):
    new_shape = [data.shape[0], data.shape[1], data.shape[2], data.shape[3],
                 n_labels]
    y = np.zeros(new_shape, np.int8)
    for label_index in range(n_labels):
        if labels is not None:
            y[:, :, :, :, label_index][
                data[:, :, :, :, 0] == labels[label_index]] = 1

    return y


def set2to0(matrix):
    matrix[matrix > 1.0] = 0.0
    return matrix


def comb_probs(prob_img: np.array, combinations: dict, root_path: str,
               labels: list):
    """ Combine probabilities according to label combinations.

    This function only combines the images, but it won't save them.

    :param prob_img: 4D image containing probability of each class in the last
    dimension.
    :param combinations: Dictionary containing the name of the combination as
    key and the combined labels as values.
    :param root_path: Where to save the combined imgs.
    :param labels: Labels that can be present in the image.
    :return: dict containing probabilities output paths as keys and combined
    imgs as values.
    """
    combined_imgs = {}
    for name, labels in combinations.items():
        result_img = np.zeros(prob_img.shape[:-1])
        saved_path = root_path + f'_{name}_prediction.nii.gz'
        for label in labels:
            pos = np.where(np.array(labels) == label)[0][0]
            result_img += prob_img[:, :, :, pos]
        combined_imgs[saved_path] = result_img
    return combined_imgs


def or_img_labels(mask_img: np.array, combinations: dict, root_path: str = '',
                  one_img: bool = False) -> Union[dict, np.ndarray]:
    """ Combine labels using OR according to label combinations dictionary.

    This function only combines the images, but it won't save them.

    :param mask_img: 3D image of the prediction.
    :param combinations: Dictionary containing the name of the combination as
    key and the combined labels as values.
    :param root_path: Where to save the combined imgs.
    :param one_img: Can be set to True if there is only one combination, so
    no dictionary will be returned but a single image with this result.
    :return: dict containing mask output paths as keys and combined imgs as
    values, or the only combined img got if one_img=True.
    """
    combined_imgs = {}
    for name, labels in combinations.items():
        saved_path = root_path + f'_{name}_mask.nii.gz' if not one_img else 'i'
        if len(labels) == 1:
            result_img = np.array(mask_img == labels[0], dtype=np.float)
        else:
            result_img = np.array(
                np.logical_or.reduce([mask_img == l for l in labels]),
                dtype=np.float)

        combined_imgs[saved_path] = result_img
    if not one_img:
        return combined_imgs  # return dict
    else:
        return combined_imgs['i']  # return only image
