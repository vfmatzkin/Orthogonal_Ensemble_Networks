import pandas as pd
import os
from configparser import ConfigParser
import seaborn as sns
from matplotlib import pyplot as plt

sns.set_style("darkgrid")


def load_metrics(base_folder):
    """ Load metrics
    loads csv files into a dataframe in folders recursively and returns a
    dictionary with the paths as keys.

    :param base_folder: Folder where to lookup the csv files.
    :return:
    """
    loaded = {}
    for root, dirs, files in os.walk(base_folder):
        if len(files) > 0:
            for f in files:
                if f.endswith('.csv'):
                    loaded[os.path.join(root, f)] = pd.read_csv(
                        os.path.join(root, f), index_col=0)
    return loaded


def load_data():
    """ Load the ini file set in the environment variable & return the metrics

    :return: loaded metrics in a dict.
    """
    ini_file = os.getenv("INI_FILE")
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

    save_ensemble_preds = parser["TEST"].getboolean(
        'save_ensemble_preds') if 'save_ensemble_preds' in parser[
        "TEST"] else False

    workspace_dir = parser['DEFAULT'].get('workspace_dir')
    segmentation_directory = parser['DEFAULT'].get(
        'segmentation_directory').replace('%workspace', workspace_dir)

    combine_labels = None if 'combine_labels' not in parser['TEST'] else \
        parser['TEST']['combine_labels']
    comb_labels = None
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

    saved_metrics = {}
    for fold in model_folds:
        datasets = [f'{dataset}_{e}' for e in
                    list(comb_labels.keys())] if comb_labels else [dataset]
        saved_metrics[fold] = load_metrics(os.path.join(out_metrics_folder,
                                                        fold))

    return saved_metrics


def organize_metrics(filename):
    """ Organize metrics: split each filename in two or three parts

    :param filename: filename (not path).
    :return: tuple containing its parts.
    """
    keyw = 'model' if 'model' in filename \
        else 'ensemble' if 'ensemble' in filename else None
    if keyw is None:
        raise ValueError(f"The filename {filename} does not contain the "
                         f"keyword 'model' or 'ensemble'")
    metric_name = filename.split(keyw)[0][:-1]
    rest = os.path.splitext(filename.split(keyw)[1])[0]
    rest = rest[1:] if 'csv' not in rest else None
    if keyw == 'ensemble' and '_' in rest:
        keyw = f"{rest.split('_')[0]}-models ensemble"
        rest = rest.split('_')[1][5:]  # remove "cross"
    if rest:
        return keyw, metric_name, rest
    else:
        return keyw, metric_name


def plot_metrics(saved_metrics, figsize=(10, 5)):
    for fold in saved_metrics.keys():
        # print(f"Fold {fold}")
        for cls in saved_metrics[fold].keys():
            filename = os.path.split(cls)[1]
            outp = organize_metrics(filename)
            # print(f"\tFile {filename} --> {outp}")
            if len(outp) == 3:
                keyw, metric_name, rest = outp
                title = f"{fold} - {metric_name} per {keyw}"  # TODO MAKE FOLD THE COLOR AND REMOVE IT FROM THE TITLE FOR HAVING UNIQUE IDS
                if not plt.fignum_exists(title):
                    plt.figure(title, figsize)
                    plt.title(title)
                    plt.xlabel("Model number")
                else:
                    plt.figure(title)
                plt.boxplot(saved_metrics[fold][cls],
                            positions=[1 + int(rest)])
            else:
                keyw, metric_name = outp
                title = f"{fold} - {metric_name} per {keyw}"
                if not plt.fignum_exists(title):
                    plt.figure(title, figsize)
                    plt.title(title)
                    plt.xlabel("Fold")
                else:
                    plt.figure(title)
                plt.boxplot(saved_metrics[fold][cls])
