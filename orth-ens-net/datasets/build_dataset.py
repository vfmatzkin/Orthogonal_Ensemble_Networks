import csv
import os
import random
import sys
from typing import Tuple

import numpy as np
from nibabel import load as load_nii

description = """build_dataset

This tool helps in converting several datasets (see below for available datase-
ts).

Usage: 
  python build_dataset.py [DATASET_NAME] [SOURCE_DIR] [OUT_PATCHES_DIR] 

IMPORTANT: Images in [SOURCE_DIR] must be organized in the following way:
- [SOURCE_DIR]
    - sub001
        - sub001_ct.nii.gz
        - sub001_seg.nii.gz
    - sub002
        - sub002_ct.nii.gz
        - sub002_seg.nii.gz

"""

DATASET_NAMES = ['Ultrecht', 'Amsterdam', 'Singapore', 'miccaibrats',
                 'hepaticvessel', 'lung']
avail = f"""Available datasets are: {', '.join(DATASET_NAMES)}."""

SUFFIXES = {'miccaibrats':
                {'images':
                     ['_flair.nii.gz', '_t1.nii.gz', '_t1ce.nii.gz',
                      '_t2.nii.gz'],
                 'labels': ['_seg.nii.gz']},
            'hepaticvessel':
                {'images': ['_ct.nii.gz'],
                 'labels': ['_seg.nii.gz']},
            'lung':
                {'images': ['_ct.nii.gz'],
                 'labels': ['_seg.nii.gz']},
            }

BATCH_SIZE = {'lung': 4}


def z_scores_normalization(img):
    print("Normalizing with z-scores...", np.shape(img))
    img = (img - np.mean(img)) / np.std(img)
    return img


NORMALIZATION = {'all': z_scores_normalization}


class Dataset:
    """ Dataset class used per storing used file/folder paths

    :param name: Dataset name. Must be one of the ones listed in DATASET_NAMES
    :param origin_dir: Source folder (contains subfolders with the images
    inside).
    :param patches_dir: Output folder (it can be inside the source folder).
    """

    def __init__(self, name=None, origin_dir=None, patches_dir=None):
        assert name in DATASET_NAMES
        assert origin_dir is not None or patches_dir is not None

        self.name = name
        self.origin_directory = origin_dir
        self.patches_directory = patches_dir

        # List the folders, but exclude the out folder if it's inside
        cond = lambda fi: os.path.join(origin_dir, fi) == patches_dir
        self.image_files = [f for f in os.listdir(origin_dir) if not cond(f)]

        self.n_images = len(self.image_files)


# Augmentation-preprocessing-matrix handling functions (not all were used yet)

def add_padding(img, depth_with_padding):
    pad_z = depth_with_padding

    pad_value = img[0][0][0]

    image_padded = np.empty((img.shape[0], img.shape[1], pad_z))
    image_padded.fill(pad_value)

    image_padded[:, :, 2:-3] = img
    return image_padded


def mask_filter_and_z_scores(background_value):
    def mask_filter_and_z_scores_background_set(img, mask):
        img[mask == 0] = np.nan  # filter from outside mask to nan
        img = (img - np.nanmean(img)) / np.nanstd(img)  # z-scores

        # filter from nan to the background value (-4 , -1 or another)
        img = np.where(mask, img, background_value)
        return img

    return mask_filter_and_z_scores_background_set


def min_max_normalization(img):
    print("Normalizing with Min-Max")  # (X = X - Xmin) / (Xmax - Xmin)
    min_img = min(img.flatten())
    max_img = max(img.flatten())
    img = (img - min_img) / (max_img - min_img)
    return img


def unsqueeze_and_concat_at_end(imgs: list):
    """ Unsqueeze images and concatenate at the end.

    Given a list of images, expand each image in the last dimension and then
    concatenate these images in the added dimension

    :param imgs: List of 3D images.
    :return: Concatenated images.
    """
    # For each modality image patch, create a separate dimension
    imgs = [np.expand_dims(d, len(d.shape)) for d in imgs]
    # Concatenate in the new dimension
    imgs = np.concatenate(imgs, len(imgs[0].shape) - 1)
    return imgs


# Unused funcs


def get_prediction_labels(prediction):
    n_samples = prediction.shape[0]
    label_arrays = []

    for sample_number in range(n_samples):
        label_data = np.argmax(prediction[sample_number], axis=0)
        label_data[label_data == 3] = 4
        label_arrays.append(np.array(label_data, dtype=np.uint8))
    return label_arrays


def get_one_hot_prediction(y_pred):
    hard_y_pred = np.copy(y_pred)

    label_data = np.argmax(y_pred, axis=4)

    for i in range(4):
        hard_y_pred[:, :, :, :, i][label_data[:, :, :, :] == i] = 1
        hard_y_pred[:, :, :, :, i][label_data[:, :, :, :] != i] = 0

    return hard_y_pred


# Patch generation functions

def generate_patches(imgs: dict, patch_size: int = 32,
                     label_id: str = '_seg.nii.gz') -> Tuple[dict, dict]:
    """  Extract patches of the supplied images according to the labels.

    :param imgs: Opened images of a patient. It is a dictionary containing the
    file ID (with extension) as keys and the numpy arrays corresponding to the
    3D image.
    :param patch_size: Size of the patches.
    :param label_id: Key used for identifying the label image that will be used
    for checking the class of the central pixel for each patch.
    :return: It returns images and labels dictionaries with classes as keys and
    lists of patches as values.
    """
    ps = patch_size // 2  # 32 --> 16
    if not label_id:
        raise AssertionError("Cannot continue: no label images provided.")
    labels = imgs[label_id]

    # Classes in the label image
    cls_labels = np.unique(labels)  # Extract the classes ([0, 1, 2])

    # Output dict {0: [[im1_t1_p1, im1_t2_p1], [im1_t1_p2, im1_t2_p2]] ..}
    # The keys are the classes and the vals lists of groups of im_ptchs.
    ptch_by_cls_imgs = {key: list() for key in cls_labels}  # Images
    ptch_by_cls_lbls = {key: list() for key in cls_labels}  # Label
    # These dicts are separated because they are saved separately afterwards

    print('------------------------------------Dentro del generate n'
          f'classes--- Label shape: {labels.shape}')
    for k in range(ps, labels.shape[2] - ps, 3):
        for i in range(ps, labels.shape[0] - ps, 4):
            for j in range(ps, labels.shape[1] - ps, 4):
                im_patches = []  # Contains a patch per image type
                lb_patches = []  # Contains a patch per label type
                for idx, img in imgs.items():  # For each image
                    if idx != label_id:  # Image (not label)
                        im_patches.append(img[i - ps:i + ps, j - ps:j + ps,
                                          k - ps:k + ps])
                    else:
                        lb_patches.append(img[i - ps:i + ps, j - ps:j + ps,
                                          k - ps:k + ps])
                center_value = labels[i][j][k]
                ptch_by_cls_imgs[center_value].append(im_patches)
                ptch_by_cls_lbls[center_value].append(lb_patches)
    return ptch_by_cls_imgs, ptch_by_cls_lbls


def write_files_metadata(dataset: Dataset, train_paths: list, val_paths: list,
                         hold_out_paths: list):
    """ Dump metadata into txt files.

    :param dataset: Dataset containing the paths of the openes files/folders.
    :param train_paths: Files used in the train split.
    :param val_paths: Files used in the validation split.
    :param hold_out_paths: Files used in the hold_out split.
    :return:
    """
    print('-------> Writing files to metadata:............................. ')
    metd_file = os.path.join(dataset.patches_directory, 'metadata_files.txt')
    ho_file = os.path.join(dataset.patches_directory, 'hold_out_files.txt')
    os.makedirs(os.path.dirname(metd_file), exist_ok=True)

    f = open(metd_file, 'w')
    for (name, paths) in [('train_paths', train_paths),
                          ('val_paths', val_paths),
                          ('hold_out_paths', hold_out_paths)]:
        f.write(name + ': \n')
        f.write(' '.join(paths))
        f.write('\n')
    f.close()
    with open(ho_file, 'w') as f:
        f.write(' '.join(hold_out_paths))
        f.write('\n')


def generate_npz_files(im_ptchs: dict, lbl_ptchs: dict, full_dirname: str,
                       images_per_file: int, drop_last: bool = True,
                       class_probs: list = None, n_file: int = 0,
                       met_no: int = 0) -> Tuple[int, int]:
    """ Generate npz files form images and labels dictionaries.

    :param im_ptchs: images dict with classes as keys and lists of patches as
    values.
    :param lbl_ptchs: labels dict with classes as keys and lists of patches as
    values.
    :param full_dirname: Out folder path
    :param images_per_file: Max amount of images per npz file.
    :param drop_last: If the last file contains less than images_per_file
    patches, don't save them.
    :param class_probs: Probability of adding a certain class patch for each
    class. If not provided, it will assign the same probability to each class.
    :param n_file: Amount of files saved so far. Used in the saved file path.
    :param met_no: Number of actual images saved, written in metadata.txt file
    and calculated as n_files x images_per_file.
    :return: Amount of saved images.
    """
    arg_classes = im_ptchs.keys()  # Classes in the provided dictionary
    if class_probs is None:
        class_probs = [1. / len(arg_classes)] * len(arg_classes)
    else:
        if len(class_probs) != len(arg_classes):
            raise AttributeError(f"You provided {len(class_probs)} "
                                 f"probabilities ({','.join(class_probs)}), "
                                 f"but {len(arg_classes)} classes were found "
                                 f"in the images ({', '.join(arg_classes)}).")

    print('Estoy generando archivos npz...')
    # Get smallest class (make sure we have at least one of each to continue).
    c, v = min([(k, len(l)) for k, l in im_ptchs.items()], key=lambda t: t[1])
    if v == 0:
        print(f'No patches for class {c}. Cannot continue')
        return n_file, met_no
    else:
        print(f'Smallest amount of im_ptchs found in class {c} ({v} im_ptchs)')
        there_are_imgs = True

    while there_are_imgs:
        images_list, labels_list = [], []
        for image_n in range(images_per_file):
            # Pick a class accordingly
            rnd_cls = np.random.choice(list(arg_classes), p=class_probs)
            if len(im_ptchs[rnd_cls]) == 0:
                print(f"No more im_ptchs for class {rnd_cls}. Stopping..")
                there_are_imgs = False
                break
            else:
                imgs, lbls = im_ptchs[rnd_cls].pop(), lbl_ptchs[rnd_cls].pop()
                imgs = unsqueeze_and_concat_at_end(imgs)
                lbls = unsqueeze_and_concat_at_end(lbls)

                images_list.append(imgs)
                labels_list.append(lbls)

        print('-------------------------------------sali del for')
        images_list = np.array(images_list, dtype=np.float32)
        labels_list = np.array(labels_list, dtype=np.float32)
        print('labels_list centers: ', labels_list[:, 16, 16, 16, 0])

        if labels_list.shape[0] < images_per_file:
            if drop_last:
                print(f"Dropping {labels_list.shape[0]} images due to they are"
                      f" less than {images_per_file}.")
                return n_file, met_no
            print(f"WARNING: saved image with {labels_list.shape[0]} images "
                  f"(less than the ims_file parameter ({images_per_file}).")
            # In the previous version, it won't save the imgs. This can be
            # counterproductive in small datasets (<1024 patches of a class).

        n_file += 1
        met_no += labels_list.shape[0]
        # ==== Saving the file
        outfile = os.path.join(full_dirname, f"archive_number_{n_file}.npz")
        os.makedirs(os.path.dirname(outfile), exist_ok=True)
        np.savez_compressed(outfile, images=images_list, labels=labels_list)
        print(outfile)

    return n_file, met_no


def generate_dataset_3d(path: str, folders_pati: list, patches_dir: str,
                        metadata_fil: str, subdirname: str, flipping: bool,
                        fixed_range: list, depth_crop: list,
                        shape_with_padding: int, dataset: str = 'miccaibrats',
                        images_per_file: int = 1024, batch_size: int = 15):
    """ Generate 3D dataset

    Given a set of images, convert them from .nii.gz to numpy array patches,
    saved as .npz files.

    :param path: Input images path.
    :param folders_pati: List of patient folders name.
    :param patches_dir: Output patches directory.
    :param metadata_fil: Filename of the metadata.
    :param subdirname: Name of the split imgs_paths.
    :param flipping: Apply flipping.
    :param fixed_range: If given, apply normalization to the supplied range.
    :param depth_crop: If given, crop the last dimension of the 3D image in the
    supplied range
    :param shape_with_padding: Pad the third dimension with the supplied value.
    :param dataset: Dataset name. Must be one of the ones listed in
    DATASET_NAMES.
    :param images_per_file: Max amount of patches stored per file.
    :param batch_size: Simultaneously subject images/patches loaded in RAM. A
    bigger number will cause more RAM usage (In hepaticvessel, a size of 20
    uses approximately 15GB of RAM).
    :return:
    """
    full_dirname = os.path.join(patches_dir, subdirname)
    print('full_dirname: ', full_dirname)
    if os.path.exists(full_dirname):
        raise RuntimeError(f'{full_dirname} already exists! Delete it if you '
                           f'want to regenerate archives.')

    suffs_img = SUFFIXES[dataset]['images']  # ['_t1.nii', '_t2.nii', ...]
    suffs_lbl = SUFFIXES[dataset]['labels']  # ['_seg1.nii', '_seg2.nii', ...]
    if len(suffs_lbl) < 1:
        raise AttributeError(f"No label suffixes provided for {dataset}. "
                             f"Add a list of labels (as strings) in "
                             f"SUFFIXES[dataset].")

    ptch_imgs, ptch_lbls, n_patches, n_files, met_no = {}, {}, {}, 0, 0
    for i, folder in enumerate(folders_pati, 1):  # For each patient's folder
        print(f"\nPatient no. {i} of {len(folders_pati)} "
              f"(Batch: {i % batch_size}/{batch_size}).")
        imgs = {}  # Here I'll store the opened images for this patient.
        for im_type in [suffs_img, suffs_lbl]:  # For images and labels
            for suff_i in im_type:  # For each suffix
                opened_file = os.path.join(path, folder, folder + suff_i)
                print('------>', opened_file)
                imgs[suff_i] = np.asanyarray(load_nii(opened_file).get_fdata())

        for suff in imgs.keys():  # For each image
            # Apply preprocessing
            if depth_crop:
                (depth_start, depth_end) = depth_crop
                print(f"Cropping depth of MRI by: [{depth_start},{depth_end}]")
                imgs[suff] = imgs[suff][:, :, depth_start:depth_end]

                if 'all' in NORMALIZATION:
                    imgs[suff] = NORMALIZATION['all'](imgs[suff])
                if dataset in NORMALIZATION:
                    imgs[suff] = NORMALIZATION[dataset](imgs[suff])

            if shape_with_padding:
                print(f"Adding padding to the first two coordinates by: "
                      f"[{shape_with_padding}]")
                imgs[suff] = add_padding(imgs[suff], shape_with_padding)

            if fixed_range:
                # Define fixed_range as follows:
                # fixed_range = {'_flair.nii.gz': [min_value, max_value],
                #                'label.nii.gz': [min_value, max_value]}
                if suff in fixed_range:
                    min_rng, max_rng = fixed_range[suff]
                    print(f"Fixing range to [{min_rng}, {max_rng}]")
                    imgs[suff] = 2. * (imgs[suff] - min_rng) / \
                                 (max_rng - min_rng) - 1

            # Apply augmentation
            if flipping:
                print('Data augmentation: flipping')
                axis_to_flip = 2
                imgs[suff] = np.flip(imgs[suff], axis_to_flip)

            # Generate im_ptchs
            print(f"Generating im_ptchs for image {folder}")

            # For the patch extractor, if there is more than one label image,
            # the first one will be used.
            label_id = suffs_lbl[0]
            ptch_im, ptch_lb = generate_patches(imgs, label_id=label_id)

            print("Patches per class: ")
            for p_class, images in ptch_im.items():  # Images
                print(f"class{p_class} im_ptchs obtained: {len(images)}.")

                if p_class not in n_patches:  # Update n_patches dict
                    n_patches[p_class] = 0
                n_patches[p_class] += len(images)

                # Add patches to the list with the other subject image im_ptchs
                if p_class not in ptch_imgs:  # New class
                    ptch_imgs[p_class] = []
                ptch_imgs[p_class] += images
            for p_class, images in ptch_lb.items():  # Labels
                if p_class not in ptch_lbls:  # New class
                    ptch_lbls[p_class] = []
                ptch_lbls[p_class] += images

            print(f"Finished generating im_ptchs from image {folder}")

        if i % batch_size == 0:
            print(f"Reached patients batch ({batch_size}). Dumping data to "
                  f"disk.")
            for p_class, images in ptch_imgs.items():
                print("Patches generated. Shuffling...")
                print(f"Class {p_class}. {len(images)} im_ptchs..")

                if len(images):
                    # Shuffling the images and im_ptchs together
                    zzip = lambda a: [list(c) for c in zip(*a)]

                    im_pt = list(zip(ptch_imgs[p_class], ptch_lbls[p_class]))
                    random.shuffle(im_pt)
                    ptch_imgs[p_class], ptch_lbls[p_class] = zzip(im_pt)

            print("Patches shuffled.")

            n_files, met_no = generate_npz_files(ptch_imgs, ptch_lbls,
                                                 full_dirname, images_per_file,
                                                 True, None, n_files, met_no)

            print("Finished saving to npz files.")
            del ptch_imgs, ptch_lbls
            ptch_imgs, ptch_lbls = {}, {}

    print("Saved to compressed npz files...")
    print("Parameters:")
    print(f"  > NUM_OF_ARCHIVES = {n_files}")
    print(f"  > NUM_IMAGES_PER_ARCHIVE = {images_per_file}")
    # WARNING: The following numbers don't consider if there were dropped
    # patches. In that case, each number will be a little smaller.
    for cls, numb in n_patches.items():
        print(f"  > NUM_CLASS{cls}_PATCHES = {numb}")

    # ==== We save the metadata
    f = open(os.path.join(patches_dir, metadata_fil), "w+")
    f.write(str(met_no))
    f.close()


def generate_train_val(dataset: Dataset, flipping: bool = False,
                       produce_hold_out: bool = True, fixed_range: list = (),
                       depth_crop: list = (), shape_with_padding: bool = None,
                       images_per_file: int = 1024, batch_size: int = 20,
                       metadata_file=None):
    """ Split the patients in train/val/holdout and produce the patches.

    :param dataset: Name of the dataset. Must be listed in DATASET_NAMES.
    :param flipping: Apply flipping.
    :param produce_hold_out: Produce holdout split.
    :param fixed_range: If given, apply normalization to the supplied range.
    :param depth_crop: If given, crop the last dimension of the 3D image in the
    supplied range
    :param shape_with_padding: Pad the third dimension with the supplied value.
    :param images_per_file: Max amount of patches stored per file.
    :param batch_size: Simultaneously subject images/patches loaded in RAM. A
    bigger number will cause more RAM usage (In hepaticvessel, a size of 20
    uses approximately 15GB of RAM).
    :param metadata_file: Use an already existing metadata_files.txt
    :return:
    """
    print('Training and validation path: ', dataset.origin_directory)
    path = dataset.origin_directory
    if metadata_file and os.path.exists(metadata_file):
        with open(metadata_file, newline='') as f:
            reader = csv.reader(f)
            files_all = list(reader)
        if not len(files_all) == 6:
            raise AssertionError(f'The metadata file must contain 6 rows, '
                                 f'train_paths, val_paths, hold_out_paths '
                                 f'and the corresponding subject names below'
                                 f'each title.')
        train_paths = files_all[1][0].strip().split(' ')
        val_paths = files_all[3][0].strip().split(' ')
        hold_out_paths = files_all[5][0].strip().split(' ')
        n_train = len(train_paths)
        n_val = len(train_paths)
    else:
        files = dataset.image_files
        random.shuffle(files)

        n_train_val = int(min(len(files) * 0.8, dataset.n_images))
        n_val = round(n_train_val * 0.06)
        print('---------------------------------n_train_val: ', n_train_val)

        if produce_hold_out:
            # In case of producing hold out (the rest of the imgs, 15% min)
            n_hold_out = max(round(len(files) * 0.2),
                             (len(files) - n_train_val))
            n_train = n_train_val - n_val
            val_paths = files[0:n_val]
            hold_out_paths = files[n_val:n_hold_out + n_val]
            train_paths = files[-n_train:]
            assert (val_paths + train_paths + hold_out_paths).sort() == \
                   files.sort()
        else:  # In case of not producing hold out
            n_train = n_train_val - n_val
            hold_out_paths = []
            train_paths = files[0:n_train]
            val_paths = files[n_train:]

        print('Writing files to metadata')
        write_files_metadata(dataset, train_paths, val_paths, hold_out_paths)

    if n_val == 0:
        print("Not enough images to produce validation batches. Be sure to "
              "move some train batches to the validation folder. Also update "
              "the metadata files accordingly.")
    else:
        print("Generating val dataset...")
        generate_dataset_3d(path, val_paths, dataset.patches_directory,
                            "metadata_val.txt", "val", flipping, fixed_range,
                            depth_crop, shape_with_padding, dataset.name,
                            images_per_file, min(n_val, batch_size))
        print(f"Finished creating val dataset on {dataset.patches_directory}.")

    # ===== We generate train and val datasets
    print("Generating train dataset...")
    generate_dataset_3d(path, train_paths, dataset.patches_directory,
                        "metadata_train.txt", "train", flipping, fixed_range,
                        depth_crop, shape_with_padding, dataset.name,
                        images_per_file, min(n_train, batch_size))
    print(f"Finished creating train dataset on {dataset.patches_directory}.")


def build_dataset(orig_dir: str, patches_dir: str,
                  dataset: str = 'miccaibrats', ims_file: int = 1024,
                  batch_size: int = 20, metd_file=None):
    """ Build dataset wrapper func.

    :param orig_dir: Input folder. Contains subfolders with the images.
    :param patches_dir: Patches output folder.
    :param dataset: Name of the dataset. Must be listed in DATASET_NAMES.
    :param ims_file: Images per npz file.
    :param batch_size: Amount of subjects used when loading images. A bigger
    number will cause more RAM usage (In hepaticvessel, a size of 20 uses
    approximately 15GB of RAM).
    :param metd_file: Previously created metadata_file.txt
    :return:
    """
    generate_train_val(Dataset(dataset, orig_dir, patches_dir),
                       images_per_file=ims_file, batch_size=batch_size,
                       metadata_file=metd_file)


if __name__ == "__main__":
    nargs = len(sys.argv)

    if nargs == 2 and sys.argv[1].lower() in ['-h', '--help']:
        print(description)
        print(avail)
        sys.exit(0)

    metd_file = None
    if nargs == 5:
        metd_file = sys.argv[4]

    if os.path.exists(sys.argv[1]) and sys.argv[1] not in DATASET_NAMES:
        raise AttributeError(f"The first parameter looks like a path, but it "
                             f"must be a dataset name. {avail}")

    dataset_name = sys.argv[1].lower() if nargs >= 2 else None
    origin_directory = sys.argv[2] if nargs >= 3 else None
    patches_directory = sys.argv[3] if nargs >= 4 else None

    batch_size = BATCH_SIZE[dataset_name] if dataset_name in BATCH_SIZE else 20
    imgs_per_file = 1024

    if not os.path.exists(origin_directory):
        raise FileNotFoundError(f"The origin directory does not exist "
                                f"({origin_directory}).")
    if not dataset_name:
        raise AttributeError(f"You must enter a valid dataset type ({avail}).")
    if not patches_directory:
        patches_directory = os.path.join(origin_directory, 'im_ptchs')
        os.makedirs(patches_directory, exist_ok=True)
        print(f"Patches dir not set. Setting it as: {patches_directory}.")

    if dataset_name not in DATASET_NAMES:
        raise AttributeError(f"Dataset {dataset_name} processing not "
                             f"implemented yet. Available datasets are: "
                             f"{avail}")
    build_dataset(origin_directory, patches_directory, dataset_name,
                  imgs_per_file, batch_size, metd_file)
