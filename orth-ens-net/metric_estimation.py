import os
import sys
from configparser import ConfigParser
from glob import glob

import nibabel as nib
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss

from utils import set2to0, ensure_dir

METRICS = ['dice', 'brier_plus', 'brier', 'variance', 'segment_variance',
           'save_ensemble_predictions']


def dice_coefficient(gt, prediction):
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
    else:  # In this case case_folder should contain model_i suffix
        probability_file = os.path.join(case_folder,
                                        f'{dataset}_prediction.nii.gz')
        probability_image = nib.load(probability_file)
        prob_img = probability_image.get_fdata()
        return prob_img


def load_segmentation(case_folder, subject_id, is_gt=False, model_name=None):
    if is_gt:
        if gt_path_structure:
            gt_file = gt_path_structure.replace('%origdir', case_folder)
            gt_file = gt_file.replace('%subject', subject_id)
        else:
            gt_file = f"{case_folder}/{subject_id}/{dataset}.nii.gz"
        gt_image = nib.load(gt_file)
        is_gt = gt_image.get_fdata()
        is_gt = set2to0(is_gt)
        return is_gt
    else:
        if model_name:
            prediction_file = os.path.join(segmentation_directory, subject_id,
                                           fold, model_name,
                                           f"{dataset}_mask.nii.gz")
        else:
            prediction_file = f"{case_folder}/{dataset}_mask.nii.gz"
        prediction_image = nib.load(prediction_file)
        prediction = prediction_image.get_fdata()
        print('Loading')
        return prediction


def brier_score(case_folder, subject_id, ensemble=False, mfiles=None):
    gt = load_segmentation(gt_directory, subject_id, True).flatten()
    probabilities = get_probabilities(case_folder, ensemble,
                                      mfiles).flatten()
    bs = brier_score_loss(gt, probabilities)
    return bs


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


def save_ensemble_pred(case_folder, model_fold, subject_id, mfiles):
    probs = get_probabilities(case_folder, True, mfiles)

    ens_pred_folder = f"./ensemble_preds/{subject_id}/{model_fold}"
    ensure_dir(ens_pred_folder)
    nib.save(nib.Nifti1Image(probs, None),
             f"{ens_pred_folder}/ensemble_{dataset}.nii.gz")

    mask_img = nib.Nifti1Image((probs > 0.5).astype(float), None)
    nib.save(mask_img, f"{ens_pred_folder}/wmh_mask.nii.gz")


def testing_models(metrics, model_fold, n_models, folder, save_results_fold):
    for metric in metrics:
        if metric not in METRICS:
            raise SystemExit(f"The metric '{metric}' does not exist")
        header = [metric]

        model_mean, model_names = list(), list()
        for model_number in range(n_models):
            rows, subject_ids = list(), list()
            model_name = f'model_{model_number}'
            model_names.append(model_name)

            for base_case_folder in glob(folder + '/*'):
                print(f"base folder: {base_case_folder}")
                subject_id = os.path.basename(base_case_folder)
                subject_ids.append(subject_id)
                case_folder = os.path.join(base_case_folder, model_fold,
                                           model_name)
                if metric == 'dice':
                    hard_dice = [dice_coefficient(
                        load_segmentation(gt_directory, subject_id, True),
                        load_segmentation(case_folder, subject_id,
                                          model_name=model_name))]
                    print('Hard_dice_anatomic: ', hard_dice)
                    rows.append(hard_dice)
                elif metric == 'brier_plus':
                    brier = [brier_plus(case_folder, subject_id)]
                    print('Brier Plus Score: ', brier)
                    rows.append(brier)
                elif metric == 'brier':
                    brier = [brier_score(case_folder, subject_id)]
                    print('Brier Score: ', brier)
                    rows.append(brier)

            df = pd.DataFrame.from_records(rows, columns=header,
                                           index=subject_ids)
            csv_folder = os.path.join('./' + save_results_fold, model_fold)
            csv_file = os.path.join(csv_folder,
                                    metric + f'_model_{model_number}.csv')
            ensure_dir(csv_folder)
            df.to_csv(csv_file)
            print(f"saved {csv_file}.")
            mean = df.mean()
            model_mean.append(mean)
            print(f"mean {mean}.")
        df_means = pd.DataFrame.from_records(model_mean, columns=header,
                                             index=model_names)
        ho_folder = os.path.join(save_results_fold, model_fold)
        os.makedirs(ho_folder, exist_ok=True)
        csv_ho = os.path.join(ho_folder, 'mean_' + metric + '_model.csv')
        df_means.to_csv(csv_ho)


def testing_ensemble(n_net, kcross, metrics, model_fold, n_models, folder,
                     save_results_fold):
    for metric in metrics:
        if metric not in METRICS:
            raise SystemExit(f"The metric '{metric}' does not exist")
        header = [metric]

        ensemble_mean, model_names = list(), list()
        models_numbers = np.arange(n_models)
        for k in range(kcross):  # 10
            np.random.shuffle(models_numbers)
            submod_numbrs = models_numbers[0:n_net]
            base_model_name = model_fold + "/model_"

            rows, subject_ids = list(), list()
            for base_case_folder in glob(folder + '/*'):
                subject_id = os.path.basename(base_case_folder)
                subject_ids.append(subject_id)
                case_folder = os.path.join(base_case_folder, base_model_name)

                if metric == 'dice':
                    hard_dice = [dice_coefficient(
                        load_segmentation(gt_directory, subject_id, True),
                        ensemble_segmentation(case_folder, submod_numbrs))]
                    rows.append(hard_dice)
                elif metric == 'brier_plus':
                    brier = [brier_plus(case_folder, subject_id, True,
                                        submod_numbrs)]
                    print('Brier Score: ', brier)
                    rows.append(brier)
                elif metric == 'brier':
                    brier = [brier_score(case_folder, subject_id, True,
                                         submod_numbrs)]
                    print('Brier Score: ', brier)
                    rows.append(brier)
                elif metric == 'variance':
                    var = [get_ensemble_variance(case_folder, submod_numbrs)]
                    print('Variance: ', var)
                    rows.append(var)
                elif metric == 'segment_variance':
                    var = [get_ensemble_variance(case_folder, submod_numbrs,
                                                 segment=True)]
                    print('Variance: ', var)
                    rows.append(var)
                elif metric == 'save_ensemble_predictions':
                    save_ensemble_pred(case_folder, model_fold, subject_id,
                                       submod_numbrs)

            model_name = metric + '_ensemble_' + str(n_net) + '_cross' + str(k)
            model_names.append(model_name)
            ensure_dir(os.path.join('../', save_results_fold, model_fold))
            df = pd.DataFrame.from_records(rows, columns=header,
                                           index=subject_ids)
            df.to_csv(os.path.join('../', save_results_fold, model_fold,
                                   model_name + '.csv'))
            ensemble_mean.append(df.mean())
        df_means = pd.DataFrame.from_records(ensemble_mean, columns=header,
                                             index=model_names)
        df_means.to_csv(os.path.join('../', save_results_fold, model_fold,
                                     'mean_' + metric + '_ensemble_' + str(
                                         n_net) + '.csv'))


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise AttributeError(f"When running 'python metric_estimation.py' you "
                             f"also must provide the path of the configuration"
                             f" file (.ini) as argument.")
    ini_file = sys.argv[1]
    parser = ConfigParser()
    parser.read(ini_file)

    workspace_dir = parser['DEFAULT'].get('workspace_dir')
    gt_directory = parser["DEFAULT"].get('image_source_dir').replace(
        '%workspace', workspace_dir)
    folder = parser["DEFAULT"].get("segmentation_directory").replace(
        '%workspace', workspace_dir)
    save_results_fold = parser["DEFAULT"].get("results_directory").replace(
        '%workspace', workspace_dir)
    model_folds = parser["ENSEMBLE"].get("pretrained_models_folds").split(",")
    n_models = parser["ENSEMBLE"].getint("n_models")

    dataset = parser["ENSEMBLE"].get("dataset")

    metrics = parser["TEST"].get("metrics").split(",")
    Nnets = parser["TEST"].get('n_net').split(',')
    kcross = parser["TEST"].getint('kcross')
    gt_path_structure = parser["TEST"].get('mask_paths')

    workspace_dir = parser['DEFAULT'].get('workspace_dir')
    segmentation_directory = parser['DEFAULT'].get(
        'segmentation_directory').replace('%workspace', workspace_dir)
    for fold in model_folds:  # XX_ResUNet_inter-orthogonal_selfp_Y_interp_Z
        testing_models(metrics, fold, n_models, folder, save_results_fold)
        for Nnet in Nnets:  # ensemble_size: 3,5  -  kcross:10
            testing_ensemble(int(Nnet), kcross, metrics, fold, n_models,
                             folder, save_results_fold)
