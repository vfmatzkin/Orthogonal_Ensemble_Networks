import os
import sys
from configparser import ConfigParser
from glob import glob

import nibabel as nib
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss

from utils import set2to0, ensure_dir


def dice_coefficient(**kwargs):
    """ Computes the Dice Coefficient between a flattened GT and Mask
    :return: Value of the DSC
    """
    return dsc(kwargs['gt'], kwargs['mask'])


def brier(**kwargs):
    """ Computes the Brier Score between a flattened GT and Mask

    :return: Value of the Brier score
    """
    return brier_score_loss(kwargs['gt'], kwargs['prob'])


def brier_plus(**kwargs):
    """ Computes the Brier Score for the pixels where the GT is 1.

    Both the GT and Mask must be flattened.

    :return: Value of the Brier Plus score
    """
    gt = kwargs['gt']
    prob = kwargs['prob']
    # Foreground voxels and probabilities
    fg_gt = gt[[gt.astype(float) == 1]]
    fg_prob = prob[[gt.astype(float) == 1]]
    bs = brier_score_loss(fg_gt, fg_prob) if len(fg_prob) > 0 else 'NaN'
    return bs


METRIC_FN = {
    'dice': dice_coefficient,
    'brier': brier,
    'brier_plus': brier_plus,
}


def dsc(gt, prediction):
    """ Sorensen–Dice coefficient.

    :param gt: Input image (X). Must be flattened.
    :param prediction: Input image (Y). Must be flattened.
    :return: Value of the coefficient.
    """
    return 2 * np.sum(gt * prediction) / (np.sum(gt) + np.sum(prediction))


def get_probabilities(case_folder, ensemble=False, mfiles=None):
    if ensemble:
        # Open the model 0 pred for getting the shape and initializing
        probability_file = os.path.join(case_folder + '0',
                                        f'{dataset}_prediction.nii.gz')
        im_shpe = (nib.load(probability_file).get_fdata()).shape
        probability_prediction = np.zeros((im_shpe[0], im_shpe[1], im_shpe[2]))

        # Get probs of all models
        for m_file in mfiles:
            probability_file = os.path.join(case_folder + str(m_file),
                                            f'{dataset}_prediction.nii.gz')
            prob_image = nib.load(probability_file).get_fdata()
            probability_prediction += prob_image[:, :, :, 0]  # Sum masks

        return np.around(probability_prediction / len(mfiles), 4)  # Divide
    # else:  # In this case case_folder should contain model_i suffix
    #     probability_file = os.path.join(case_folder,
    #                                     f'{dataset}_prediction.nii.gz')
    #     probability_image = nib.load(probability_file)
    #     prob_img = probability_image.get_fdata()
    #     return prob_img


# def load_segmentation(case_folder, subject_id, is_gt=False, model_name=None):
# if is_gt:
#     if gt_path_str:
#         gt_file = gt_path_str.replace('%origdir', case_folder)
#         gt_file = gt_file.replace('%subject', subject_id)
#     else:
#         gt_file = f"{case_folder}/{subject_id}/{dataset}.nii.gz"
#     gt_image = nib.load(gt_file)
#     is_gt = gt_image.get_fdata()
#     is_gt = set2to0(is_gt)
#     return is_gt
# else:
#     if model_name:
#         prediction_file = os.path.join(segmentation_directory, subject_id,
#                                        fold, model_name,
#                                        f"{dataset}_mask.nii.gz")
#     else:
#         prediction_file = f"{case_folder}/{dataset}_mask.nii.gz"
#     prediction_image = nib.load(prediction_file)
#     prediction = prediction_image.get_fdata()
#     print('Loading')
#     return prediction


def brier_plus(case_folder, subject_id, ensemble=False, mfiles=None):
    gt = load_segmentation(gt_directory, subject_id, True).flatten()
    foreground_voxels = gt[[gt.astype(float) == 1]]
    probabilities = get_probabilities(case_folder, ensemble,
                                      mfiles).flatten()
    foreground_probabilities = probabilities[[gt.astype(float) == 1]]
    if len(foreground_probabilities) > 0:
        bs = brier_score_loss(foreground_voxels, foreground_probabilities)
    else:
        bs = 'NaN'
    return bs


def get_ensemble_variance(case_folder, mfiles=None, segment=False):
    ensemble_mean_prob = get_probabilities(case_folder, True, mfiles)
    if segment:
        ensemble_lm = (ensemble_mean_prob > 0.5).astype(
            float)
        positive_pixels = (ensemble_lm == 1).astype(float)
    variance_estimation = 0

    for mfile in mfiles:
        probability_file = os.path.join(case_folder + str(mfile),
                                        f'{dataset}_prediction.nii.gz')
        prob_image = nib.load(probability_file).get_fdata()[:, :, :, 0]
        variance_image = np.square(prob_image - ensemble_mean_prob)

        if segment:
            variance = np.sum(variance_image * positive_pixels) / np.sum(
                positive_pixels)
        else:
            variance = np.mean(variance_image)
        variance_estimation += variance

    return variance_estimation / len(mfiles)


def ensemble_segmentation(case_folder, m):
    ensemble_probs = get_probabilities(case_folder, True, m)
    ensemble_label_map = (ensemble_probs > 0.5).astype(float)
    return ensemble_label_map


def save_ensemble_pred(prob_image, mask_image, gt_image, k, ensemble_size,
                       subject_id, root_folder, inp_size):
    ens_name = f'ensemble_{ensemble_size}_cross{k}'
    preds_folder = os.path.join(root_folder, ens_name, subject_id)
    ensure_dir(preds_folder)

    prob_path = os.path.join(preds_folder, f'{dataset}_prediction.nii.gz')
    mask_path = os.path.join(preds_folder, f'{dataset}_mask.nii.gz')
    gt_path = os.path.join(preds_folder, f'{dataset}_gt.nii.gz')

    nib.save(nib.Nifti1Image(prob_image.reshape(inp_size), None), prob_path)
    nib.save(nib.Nifti1Image(mask_image.reshape(inp_size), None), mask_path)
    nib.save(nib.Nifti1Image(gt_image.reshape(inp_size), None), gt_path)


def load_image(path: str, labels: list = None) -> np.ndarray:
    """ Load image using nibabel and return its float numpy array.

    :param path: Input image path.
    :param labels: Labels used during training. Used for masking the loaded
    image.
    :return: Image cast as array.
    """
    gt_image = nib.load(path)
    img = gt_image.get_fdata()
    out_img = np.zeros_like(img)
    if labels:  # Only grab the used labels
        for label in labels:
            out_img += (img == label) * label
    else:
        out_img = img

    return out_img


def test_models(metrics: list, model_fold: str, n_models: int,
                predictions_folder: str, out_metrics_folder: str):
    """ Calculate metrics on holdout split for all the models of a fold.

    Given a trained fold, calculate the requested metrics for all the trained
    models.

    :param metrics: List containing the metrics to calculate (dice, brier,
    brier_plus)
    :param model_fold: Trained fold (e.g.: XX_ResUNet_inter-orthogonal_selfp_X)
    :param n_models: Amount of trained models.
    :param predictions_folder: Predictions folder. Contains subfolders for each
    patient.
    :param out_metrics_folder: Path of the output folder for the metrics.
    """
    print("Testing models")
    # Folder where the csv files will be saved
    csv_folder = os.path.join(out_metrics_folder, model_fold, 'models')
    ensure_dir(csv_folder)

    print(f"Metrics to test: {', '.join(metrics)}.")
    for metric in metrics:  # dice, brier
        if metric not in METRIC_FN:
            raise SystemExit(f"The metric '{metric}' does not exist")
        header = [metric]
        print(f"\nMetric: {metric}.\n")

        model_mean, model_names = list(), list()
        for model_number in range(n_models):  # From model 0 to model 10
            print(f"Model {model_number}/{n_models}:")
            rows, subject_ids = list(), list()
            model_name = f'model_{model_number}'
            model_names.append(model_name)

            print("  subjects: ", end='')
            for pred_subject in glob(predictions_folder + '/*'):  # Pred folder
                subject_id = os.path.basename(pred_subject)
                print(f"{subject_id}", end=' ')
                subject_ids.append(subject_id)

                preds_folder = os.path.join(segmentation_directory, subject_id,
                                            fold, model_name)

                # Images paths
                mask_path = os.path.join(preds_folder,
                                         f"{dataset}_mask.nii.gz")
                prob_path = os.path.join(preds_folder,
                                         f"{dataset}_prediction.nii.gz")
                gt_path = gt_path_str.replace('%subject', subject_id)

                gt_image = load_image(gt_path, labels).flatten()
                mask_image = load_image(mask_path).flatten()
                prob_image = load_image(prob_path).flatten()

                result = [METRIC_FN[metric](gt=gt_image, mask=mask_image,
                                            prob=prob_image)]
                rows.append(result)

            # Dataframe contains each subject metric
            subjects_model = pd.DataFrame.from_records(rows, columns=header,
                                                       index=subject_ids)

            subjects_data_fname = metric + f'_model_{model_number}.csv'
            subjects_data_fpath = os.path.join(csv_folder, subjects_data_fname)
            subjects_model.to_csv(subjects_data_fpath)
            print(f"\n  saved {subjects_data_fpath}.")

            model_mean.append(subjects_model.mean())
            print(f"  mean {float(model_mean[-1])}.")
        models_means = pd.DataFrame.from_records(model_mean, columns=header,
                                                 index=model_names)
        mean_metric_per_model = os.path.join(csv_folder,
                                             'mean_' + metric + '_model.csv')
        models_means.to_csv(mean_metric_per_model)
        print(f"  means saved in {mean_metric_per_model}")


def load_ensemble_images(ens_models_paths: list, flatten=True,
                         get_size=False) -> tuple:
    """ Load ensemble images

    Given a list of paths of the models corresponding to an ensemble, merge the
    probabilities and the masks for these models.

    :param ens_models_paths: List containing the paths corresponding to the
    ensemble.
    :param flatten: Flatten the images before returning them.
    :param get_size: Return also the input image size (useful when saving ensem
    ble predictions).
    :return: (mask_image, prob_image) merged mask and probability images, as a
    tuple.
    """
    prob_imgs, mask_imgs = [], []
    for model_folder in ens_models_paths:  # Load the imgs to the list
        prob_path = os.path.join(model_folder, f'{dataset}_prediction.nii.gz')
        mask_path = os.path.join(model_folder, f'{dataset}_mask.nii.gz')
        prob_imgs.append(load_image(prob_path))
        mask_imgs.append(load_image(mask_path))
    prob_img = np.around(np.mean(prob_imgs, axis=0), 4)
    mask_img = np.array(np.mean(prob_imgs) > 0.5, dtype=prob_img.dtype)  # TODO THIS IS NOT IN THE ORIG SIZE
    inp_size = prob_img.shape if get_size else None
    if flatten:
        mask_img, prob_img = mask_img.flatten(), prob_img.flatten()
    if get_size:
        return mask_img, prob_img, inp_size
    return mask_img, prob_img


def test_ensembles(ensemble_size: int, k_cross: int, metrics: str,
                   model_fold: str, n_models: int, predictions_folder: str,
                   out_metrics_folder: str):
    """

    :param ensemble_size: Size of the ensembles.
    :param k_cross: Amount of ensembles to randomly group.
    :param metrics: List containing the metrics to calculate (dice, brier,
    brier_plus)
    :param model_fold: Trained fold (e.g.: XX_ResUNet_inter-orthogonal_selfp_X)
    :param n_models: Amount of trained models.
    :param predictions_folder: Predictions folder. Contains subfolders for each
    patient.
    :param out_metrics_folder: Path of the output folder for the metrics.
    :return:
    """
    # Folder where the csv files will be saved
    print("Testing ensembles")
    csv_folder = os.path.join(out_metrics_folder, model_fold, 'ensembles')
    ensure_dir(csv_folder)

    print(f"Metrics to test: {', '.join(metrics)}.")
    print(f"{k_cross} ensembles of size {ensemble_size} will be formed ("
          f"{n_models} models available).")
    ens_means, model_names = [], []
    models_numbers = np.arange(n_models)
    for k in range(k_cross):
        print(f"\nEnsemble {k}/{k_cross} (size={ensemble_size})")
        np.random.shuffle(models_numbers)
        submod_numbrs = models_numbers[0:ensemble_size]

        for metric in metrics:
            if metric not in METRIC_FN:
                raise SystemExit(f"The metric '{metric}' does not exist")
            header = [metric]
            print(f"Metric: {metric}.")
            rows, subject_ids = list(), list()

            print("  subjects: ", end='')
            for pred_subject in glob(predictions_folder + '/*'):
                subject_id = os.path.basename(pred_subject)
                print(f"{subject_id}", end=' ')
                subject_ids.append(subject_id)

                preds_folder = os.path.join(segmentation_directory, subject_id,
                                            fold)
                ens_models_paths = [os.path.join(preds_folder, f'model_{m}')
                                    for m in submod_numbrs]
                gt_path = gt_path_str.replace('%subject', subject_id)

                # Load the mask and prediction of the ensemble & the GT
                mask_image, prob_image, inp_size = load_ensemble_images(
                    ens_models_paths, flatten=True, get_size=True)
                gt_image = load_image(gt_path, labels).flatten()

                if metric == metrics[-1] and save_ensemble_preds:  # Last mtric
                    save_ensemble_pred(prob_image, mask_image, gt_image, k,
                                       ensemble_size, subject_id, csv_folder,
                                       inp_size)

                result = [METRIC_FN[metric](gt=gt_image, mask=mask_image,
                                            prob=prob_image)]

                rows.append(result)

            model_file_name = f'{metric}_ensemble_{ensemble_size}_cross{k}.csv'
            model_names.append(os.path.splitext(model_file_name)[0])

            # Dataframe contains metric for each subj that particular ensemble
            metr_ens_sub = pd.DataFrame.from_records(rows, subject_ids,
                                                     columns=header)
            ens_sub_fpath = os.path.join(csv_folder, model_file_name)
            metr_ens_sub.to_csv(ens_sub_fpath)
            print(f"\n  saved {ens_sub_fpath}.")
            ens_means.append(metr_ens_sub.mean())  # Save mean
            print(f"  mean {float(ens_means[-1])}.")

    # Dataframe contains means for each ensemble
    met_ens_mean = pd.DataFrame.from_records(ens_means, model_names,
                                             columns=header)
    met_ens_mean_fname = f'{metric}_ensemble_{ensemble_size}_cross{k}.csv'
    met_ens_mean_fpath = os.path.join(csv_folder, met_ens_mean_fname)

    met_ens_mean.to_csv(met_ens_mean_fpath)
    print(f"  means saved in {met_ens_mean_fpath}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise AttributeError(f"When running 'python metric_estimation.py' you "
                             f"also must provide the path of the configuration"
                             f" file (.ini) as argument.")
    ini_file = sys.argv[1]
    parser = ConfigParser()
    parser.read(ini_file)

    if not os.path.exists(ini_file):
        raise FileNotFoundError(f"The configuration file ({ini_file}) could "
                                f"not be found. Make sure you provide its full"
                                f" path.")

    workspace_dir = parser['DEFAULT'].get('workspace_dir')
    gt_directory = parser["DEFAULT"].get('image_source_dir').replace(
        '%workspace', workspace_dir)
    preds_folder = parser["DEFAULT"].get(
        "segmentation_directory").replace('%workspace', workspace_dir)
    out_metrics_folder = parser["DEFAULT"].get("results_directory").replace(
        '%workspace', workspace_dir)
    model_folds = parser["ENSEMBLE"].get("pretrained_models_folds").split(",")
    n_models = parser["ENSEMBLE"].getint("n_models")

    dataset = parser["ENSEMBLE"].get("dataset")

    out_channels = parser["TRAIN"].getint("output_channels")
    if 'labels' in parser['TRAIN']:  # labels='1,2,4' --> labels = [1, 2, 4]
        lab = parser['TRAIN']['labels']
        labels = list(map(int, lab.split(','))) if ',' in lab else lab
    else:
        labels = list(range(out_channels))  # out_ch=3 -> labels = [0,1,2]

    metrics = parser["TEST"].get("metrics").split(",")
    ensemble_sizes = parser["TEST"].get('Nnet').split(',')
    k_cross = parser["TEST"].getint('kcross')

    gt_path_str = parser["TEST"].get('mask_paths')
    gt_path_str = gt_path_str.replace('%origdir', gt_directory)

    save_ensemble_preds = parser["TEST"].getboolean('save_ensemble_preds') \
        if 'save_ensemble_preds' in parser["TEST"] else False

    workspace_dir = parser['DEFAULT'].get('workspace_dir')
    segmentation_directory = parser['DEFAULT'].get(
        'segmentation_directory').replace('%workspace', workspace_dir)
    for fold in model_folds:  # XX_ResUNet_inter-orthogonal_selfp_Y_interp_Z
        # test_models(metrics, fold, n_models, preds_folder, out_metrics_folder)
        for ensemble_size in ensemble_sizes:  # n_net: 3,5  -  kcross:10
            test_ensembles(int(ensemble_size), k_cross, metrics, fold,
                           n_models, preds_folder, out_metrics_folder)
