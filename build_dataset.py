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

import click
import os
import random
import sys
from pdb import set_trace as st
from random import randint

import numpy as np
from nibabel import load as load_nii

DATASET_NAMES = ['Ultrecht', 'Amsterdam', 'Singapore', 'miccaibrats',
                 'hepaticvessel']
avail = f"""Available datasets are: {', '.join(DATASET_NAMES)}."""

SUFFIXES = {'miccaibrats':
                {'images':
                     ['_flair.nii.gz', '_t1.nii.gz', '_t1ce.nii.gz',
                      '_t2.nii.gz'],
                 'labels': ['_seg.nii.gz']},
            'hepaticvessel':
                {'images': ['_ct.nii.gz'],
                 'labels': ['_seg.nii.gz']},
            }


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


def z_scores_normalization(img):
    print("Normalizing with z-scores...", np.shape(img))
    img = (img - np.mean(img)) / np.std(img)
    return img


def min_max_normalization(img):
    print("Normalizing with Min-Max")  # (X = X - Xmin) / (Xmax - Xmin)
    min_img = min(img.flatten())
    max_img = max(img.flatten())
    img = (img - min_img) / (max_img - min_img)
    return img


class Normalization:
    def __init__(self, method, background_value=None):
        if method == "z_scores":
            self.takes_mask = False
            self.method = z_scores_normalization

        elif method == "mask_filter_and_z_scores":
            assert background_value is not None
            self.takes_mask = True
            self.method = mask_filter_and_z_scores(background_value)

        elif method == "min_max":
            self.takes_mask = False
            self.method = min_max_normalization

        else:
            raise RuntimeError("Unknown normalization method.")


class Dataset:
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


def print_parameters(num_of_archives, num_images_per_archive,
                     num_class1_patches, num_class2_patches,
                     num_class4_patches, num_negative_patches):
    print("Saving to compressed npz files...")
    print("Parameters:")
    print(f"  > NUM_OF_ARCHIVES = {num_of_archives}")
    print(f"  > NUM_IMAGES_PER_ARCHIVE = {num_images_per_archive}")
    print(f"  > NUM_CLASS1_PATCHES = {num_class1_patches}")
    print(f"  > NUM_CLASS2_PATCHES = {num_class2_patches}")
    print(f"  > NUM_CLASS4_PATCHES = {num_class4_patches}")
    print(f"  > NUM_NEGATIVE_PATCHES = {num_negative_patches}")


def generate_patch_4_classes(elem):
    class1_patches_single_image, class2_patches_single_image, \
    class4_patches_single_image, negative_patches_single_image = [], [], [], []
    flair = elem["flair"]
    t1 = elem["t1"]
    t2 = elem["t2"]
    t1ce = elem["t1ce"]
    labels = elem["labels"]

    assert t1.shape == flair.shape
    assert t1.shape == labels.shape
    assert t1.shape == t2.shape
    assert t1.shape == t1ce.shape
    print('------------------------------------Dentro del generate 4 '
          f'classes--- Label shape: {labels.shape}')
    for k in range(16, labels.shape[2] - 16, 3):
        # k is the number of image----> Agos: k no es el numero de imagenes,
        # sino la tercera dimensión
        for i in range(16, labels.shape[0] - 16, 4):
            for j in range(16, labels.shape[1] - 16, 4):
                # (i,j) is the center

                new_patch_labels = labels[i - 16:i + 16, j - 16:j + 16,
                                   k - 16:k + 16]
                new_patch_flair = flair[i - 16:i + 16, j - 16:j + 16,
                                  k - 16:k + 16]
                new_patch_t1 = t1[i - 16:i + 16, j - 16:j + 16, k - 16:k + 16]
                new_patch_t2 = t2[i - 16:i + 16, j - 16:j + 16, k - 16:k + 16]
                new_patch_t1ce = t1ce[i - 16:i + 16, j - 16:j + 16,
                                 k - 16:k + 16]

                if labels[i][j][k] == 1.0:
                    class1_patches_single_image.append(
                        [new_patch_t1, new_patch_t1ce, new_patch_t2,
                         new_patch_flair, new_patch_labels])

                elif labels[i][j][k] == 2.0:
                    class2_patches_single_image.append(
                        [new_patch_t1, new_patch_t1ce, new_patch_t2,
                         new_patch_flair, new_patch_labels])
                elif labels[i][j][k] == 4.0:
                    class4_patches_single_image.append(
                        [new_patch_t1, new_patch_t1ce, new_patch_t2,
                         new_patch_flair, new_patch_labels])
                else:
                    negative_patches_single_image.append(
                        [new_patch_t1, new_patch_t1ce, new_patch_t2,
                         new_patch_flair, new_patch_labels])
    return class1_patches_single_image, class2_patches_single_image, \
           class4_patches_single_image, negative_patches_single_image


def generate_patch_4_classes_ref(imgs, patch_size=32, label_id='_seg.nii.gz'):
    """  Extract im_ptchs of the supplied images according to the labels.

    :param imgs:
    :param patch_size:
    :param label_id:
    :return:
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


def write_files_metadata(dataset, train_paths, val_paths, hold_out_paths):
    print('-------> Writing files to metadata:............................. ')
    metadata_file = os.path.join(dataset.patches_directory,
                                 "metadata_files.txt")
    os.makedirs(os.path.dirname(metadata_file), exist_ok=True)

    f = open(metadata_file, "w")
    for (name, paths) in [("train_paths", train_paths),
                          ("val_paths", val_paths),
                          ("hold_out_paths", hold_out_paths)]:
        f.write(name + ": \n")
        for elem in paths:
            f.write(elem + " ")
        f.write("\n")
    f.close()


def generate_npz_files(class1_patches, class2_patches, class4_patches,
                       negative_patches, num_images_per_archive, full_dirname,
                       num_of_archives=None):
    # This function generates npz files from positive and negative im_ptchs
    # We could flip the im_ptchs on the edge Y if necessary for augmentation -
    # but gets too noisy
    if num_of_archives is None:
        num_of_archives = 0  # total number of archives generated
    num_of_archive = num_of_archives - 1  # number of each archive
    print('Estoy generando archivos npz y el numero de archivos es: ',
          num_of_archive)
    # ===== We generate npz files until we have no more im_ptchs
    while 1:
        num_of_archive += 1
        t1_flair_list, labels_list = [], []
        for num_image in range(num_images_per_archive):

            # --Damos un 25% de posibilidades a que el voxel central pertenezca
            # a cada una de las 4 clases
            rand = randint(0, 100)
            if rand <= 25:
                if len(class1_patches) == 0:
                    print(
                        'Dejo de archivar porque me quede sin archivos de '
                        'clase 1.')
                    # No more positive im_ptchs, dataset is done
                    return num_of_archives
                data = class1_patches.pop()
            elif 25 < rand <= 50:
                if len(class2_patches) == 0:
                    # No more positive im_ptchs, dataset is done
                    print('Dejo de archivar porque me quede sin archivos de '
                          'clase 2.')
                    return num_of_archives
                data = class2_patches.pop()
            elif 50 < rand <= 75:
                if len(class4_patches) == 0:
                    print(
                        'Dejo de archivar porque me quede sin archivos de '
                        'clase 4')
                    # No more positive im_ptchs, dataset is done
                    return num_of_archives
                data = class4_patches.pop()
            else:
                if len(negative_patches) == 0:
                    # No more negative im_ptchs, dataset is done
                    print(
                        'Dejo de archivar porque me quede sin archivos de '
                        'clase 0')
                    return num_of_archives
                data = negative_patches.pop()
            try:
                t1, t1ce, t2, flair, labels = data
                # ===== we mix t1 and flair
                new_shape_t1_flair_joined = (
                    4, t1.shape[0], t1.shape[1], t1.shape[2])
                t1_flair_joined = np.empty(new_shape_t1_flair_joined)
                t1_flair_joined[0] = t1
                t1_flair_joined[1] = t1ce
                t1_flair_joined[2] = t2
                t1_flair_joined[3] = flair

                # we swap the axes (4,32,32,32) => (32,4,32,32) =>
                # (32,32,4,32) => (32,32,32,4)
                t1_flair_joined = np.swapaxes(
                    np.swapaxes(np.swapaxes(t1_flair_joined, 0, 1), 1, 2), 2,
                    3)

                # ===== we produce one label matrix, which is the probability
                # of WMH.
                new_shape_one_labels = (
                    1, labels.shape[0], labels.shape[1], labels.shape[2])
                one_label_joined = np.empty(new_shape_one_labels)
                one_label_joined[0] = labels  # probability of WMH

                # ===== we swap the axes (1,32,32,32) => (32,1,32,32) =>
                # (32,32,1,32) => (32,32,32,1)
                one_label_joined = np.swapaxes(
                    np.swapaxes(np.swapaxes(one_label_joined, 0, 1), 1, 2), 2,
                    3)

                # ===== we append them
                t1_flair_list.append(t1_flair_joined)
                labels_list.append(one_label_joined)

            except:
                raise RuntimeError(f"Image broken {data}")

        print('-------------------------------------sali del for')
        t1_flair_list = np.array(t1_flair_list, dtype=np.float32)
        labels_list = np.array(labels_list, dtype=np.float32)
        print('labels_list centers: ', labels_list[:, 16, 16, 16, 0])
        assert t1_flair_list.shape == (1024, 32, 32, 32, 4)

        # ==== Saving the file
        outfile = os.path.join(full_dirname,
                               f"archive_number_{num_of_archive}.npz")
        os.makedirs(os.path.dirname(outfile), exist_ok=True)
        np.savez_compressed(outfile, t1_flair=t1_flair_list,
                            labels=labels_list)
        print(outfile)
        num_of_archives += 1

        return num_of_archives


def unsqueeze_and_concat_at_end(imgs):
    # For each modality image patch, create a separate dimension
    imgs = [np.expand_dims(d, len(d.shape)) for d in imgs]
    # Concatenate in the new dimension
    imgs = np.concatenate(imgs, len(imgs[0].shape) - 1)
    return imgs


def generate_npz_files_ref(im_ptchs, lbl_ptchs, images_per_file,
                           full_dirname, class_probs=None,
                           dataset='miccaibrats', n_file=0):
    """ Generate npz files form im_patches dictionary

    :param im_ptchs:
    :param lbl_ptchs:
    :param images_per_file:
    :param full_dirname:
    :param class_probs:
    :param n_file:
    :return:
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
    # Get smallest class
    c, v = min([(k, len(l)) for k, l in im_ptchs.items()], key=lambda t: t[1])
    if v == 0:
        print(f'No patches for class {c}. Cannot continue')
        return 0
    else:
        print(f'Smallest amount of im_ptchs found in class {c} ({v} im_ptchs)')
        there_are_imgs = True

    while there_are_imgs:
        n_file += 1
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
            print(f"WARNING: saved image with {labels_list.shape[0]} images (less "
                  f"than the images_per_file parameter ({images_per_file}).")
            # In the previous version, it won't save the imgs. This can be
            # counterproductive in small datasets (<1024 patches of a class).

        # ==== Saving the file
        outfile = os.path.join(full_dirname, f"archive_number_{n_file}.npz")
        os.makedirs(os.path.dirname(outfile), exist_ok=True)
        np.savez_compressed(outfile, images=images_list, labels=labels_list)
        print(outfile)

    return n_file


def generate_dataset_3d(path, files_paths, patches_dir, metadata_dir,
                        subdirname, normalization, boxcox_lambda, flipping,
                        fixed_range, depth_crop, shape_with_padding,
                        mask_filtering):
    full_dirname = os.path.join(patches_dir, subdirname)
    print('full_dirname: ', full_dirname)
    if os.path.exists(full_dirname):
        raise RuntimeError(f'{full_dirname} already exists! Delete it if you '
                           f'want to regenerate archives.')

    one_element = {}
    elem_number = 0
    class1_patches = []
    class2_patches = []
    class4_patches = []
    negative_patches = []
    num_of_archives = 0
    for file_path in files_paths:
        num_class1_patches = len(class1_patches)
        num_class2_patches = len(class2_patches)
        num_class4_patches = len(class4_patches)
        num_negative_patches = len(negative_patches)

        print(f"> NUM POSITIVE CLASS1: {num_class1_patches}")
        print(f"> NUM POSITIVE CLASS2: {num_class2_patches}")
        print(f"> NUM POSITIVE CLASS4: {num_class4_patches}")
        print(f"> NUM NEGATIVE PATCHES: {num_negative_patches}")

        # ===== We generate the im_ptchs from the loaded data ----- NUEVO I
        if elem_number == 30:
            print("Generating im_ptchs...")
            class1_patches = []
            class2_patches = []
            class4_patches = []
            negative_patches = []
            elem_number = 0
        # ===== We generate the im_ptchs from the loaded data ----- NUEVO F

        print('------>',
              os.path.join(path, file_path, file_path + '_flair.nii.gz'))
        flair_nii_gz = load_nii(
            os.path.join(path, file_path,
                         file_path + '_flair.nii.gz')).get_data()

        t1_nii_gz = load_nii(
            os.path.join(path, file_path, file_path + '_t1.nii.gz')).get_data()

        t1ce_nii_gz = load_nii(
            os.path.join(path, file_path,
                         file_path + '_t1ce.nii.gz')).get_data()

        t2_nii_gz = load_nii(
            os.path.join(path, file_path, file_path + '_t2.nii.gz')).get_data()

        labels_nii_gz = load_nii(
            os.path.join(path, file_path,
                         file_path + '_seg.nii.gz')).get_data()

        # Como estamos en un problema multiclase le saco la parte de set2 to0
        try:
            assert t1_nii_gz.shape == flair_nii_gz.shape
            assert t1_nii_gz.shape == labels_nii_gz.shape
        except:
            st()

        brainmask_nii_gz = None
        if depth_crop is not None:
            (depth_start, depth_end) = depth_crop
            print(f"Cropping depth of MRI by: [{depth_start},{depth_end}]")
            flair_nii_gz = flair_nii_gz[:, :, depth_start:depth_end]
            t1_nii_gz = t1_nii_gz[:, :, depth_start:depth_end]
            labels_nii_gz = labels_nii_gz[:, :, depth_start:depth_end]
            brainmask_nii_gz = brainmask_nii_gz[:, :, depth_start:depth_end]

        # Eliminé la clase porque no anda y directamente trabajo con el método
        # z-score
        if normalization is not None:
            t1_nii_gz = normalization(t1_nii_gz)
            flair_nii_gz = normalization(flair_nii_gz)
            t1ce_nii_gz = normalization(t1ce_nii_gz)
            t2_nii_gz = normalization(t2_nii_gz)

        if shape_with_padding is not None:
            flair_nii_gz = add_padding(flair_nii_gz, shape_with_padding)
            t1_nii_gz = add_padding(t1_nii_gz, shape_with_padding)
            labels_nii_gz = add_padding(labels_nii_gz, shape_with_padding)
            brainmask_nii_gz = add_padding(brainmask_nii_gz,
                                           shape_with_padding)
            print(f"Adding padding to the first two coordinates by: "
                  f"[{shape_with_padding}]")

        try:
            assert t1_nii_gz.shape == flair_nii_gz.shape
            assert t1_nii_gz.shape == labels_nii_gz.shape
            assert t1_nii_gz.shape == t1ce_nii_gz.shape
            assert t1_nii_gz.shape == t2_nii_gz.shape
        except:
            st()

        if fixed_range is not None:
            (t1_min_voxel_value, t1_max_voxel_value, flair_min_voxel_value,
             flair_max_voxel_value) = fixed_range
            print("Fixing range to [-1,1]")
            # === we modify the range to [-1,1]
            t1_nii_gz = 2. * (t1_nii_gz - t1_min_voxel_value) / (
                    t1_max_voxel_value - t1_min_voxel_value) - 1
            flair_nii_gz = 2. * (flair_nii_gz - flair_min_voxel_value) / (
                    flair_max_voxel_value - flair_min_voxel_value) - 1

        one_element["flair"] = flair_nii_gz
        one_element["t1"] = t1_nii_gz
        one_element["labels"] = labels_nii_gz
        one_element["t2"] = t2_nii_gz
        one_element["t1ce"] = t1ce_nii_gz

        # ===== We flip the images on the Y edge for data augmentation
        data_augmentation_by_flipping = flipping
        print('Data augmentation: ', data_augmentation_by_flipping)
        if data_augmentation_by_flipping:
            axis_to_flip = 2
            one_element_flip = []
            flair_nii_gz_flip = np.flip(flair_nii_gz, axis_to_flip)
            t1_nii_gz_flip = np.flip(t1_nii_gz, axis_to_flip)
            labels_nii_gz_flip = np.flip(labels_nii_gz, axis_to_flip)

            one_element_flip.append(flair_nii_gz_flip)
            one_element_flip.append(t1_nii_gz_flip)
            one_element_flip.append(labels_nii_gz_flip)

        # ===== We add them to the labeled data
        print(f"Generating im_ptchs from image N°{elem_number}")

        class1_patches_single_image, class2_patches_single_image, \
        class4_patches_single_image, \
        negative_patches_single_image = generate_patch_4_classes(one_element)
        # Agos: im_ptchs positivos son los que tienen algunas de las 3 clases
        # como pixel central

        print(f"class1_patches obtained: {len(class1_patches_single_image)}  |"
              f" class2_patches obtained: {len(class2_patches_single_image)}  "
              f"| class4_patches obtained: {len(class4_patches_single_image)} "
              f" | Negative im_ptchs obtained: "
              f"{len(negative_patches_single_image)}")

        class1_patches = class1_patches + class1_patches_single_image
        class2_patches = class2_patches + class2_patches_single_image
        class4_patches = class4_patches + class4_patches_single_image
        negative_patches = negative_patches + negative_patches_single_image
        print(f"class1_patches total: {len(class1_patches)}  | class2_patches "
              f"obtained: {len(class2_patches)}  | class4_patches obtained: "
              f"{len(class4_patches)}  | Negative im_ptchs obtained: "
              f"{len(negative_patches)}")

        print(f"Finished generating im_ptchs from image N°{elem_number}")
        elem_number = elem_number + 1

        print("Data loaded.")

        if elem_number == 30:
            # ==== We shuffle the im_ptchs
            print("Patches generated. Shuffling...")
            random.shuffle(class1_patches)
            random.shuffle(class2_patches)
            random.shuffle(class4_patches)
            random.shuffle(negative_patches)
            print("Patches shuffled.")

            # ===== Parameters
            num_images_per_archive = 1024
            num_class1_patches = len(class1_patches)
            num_class2_patches = len(class2_patches)
            num_class4_patches = len(class4_patches)
            num_negative_patches = len(negative_patches)

            print(f"> NUM POSITIVE CLASS1: {num_class1_patches}")
            print(f"> NUM POSITIVE CLASS2: {num_class2_patches}")
            print(f"> NUM POSITIVE CLASS4: {num_class4_patches}")
            print(f"> NUM NEGATIVE PATCHES: {num_negative_patches}")

            num_of_archives = generate_npz_files(class1_patches,
                                                 class2_patches,
                                                 class4_patches,
                                                 negative_patches,
                                                 num_images_per_archive,
                                                 full_dirname, num_of_archives)

            print_parameters(num_of_archives, num_images_per_archive,
                             num_class1_patches, num_class2_patches,
                             num_class4_patches, num_negative_patches)

            # ==== We save the metadata
            f = open(os.path.join(patches_dir, metadata_dir), "w+")
            f.write(str(num_of_archives * num_images_per_archive))
            f.close()
            print("Finished saving to npz files.")

    # ==== We shuffle the im_ptchs
    print("Ultimos parches generados......Patches generated. Shuffling...")
    print('element number.....', elem_number)
    random.shuffle(class1_patches)
    random.shuffle(class2_patches)
    random.shuffle(class4_patches)
    random.shuffle(negative_patches)
    print("Patches shuffled.")

    # ===== Parameters
    num_images_per_archive = 1024
    num_class1_patches = len(class1_patches)
    num_class2_patches = len(class2_patches)
    num_class4_patches = len(class4_patches)
    num_negative_patches = len(negative_patches)

    print(f"> NUM POSITIVE CLASS1: {num_class1_patches}")
    print(f"> NUM POSITIVE CLASS2: {num_class2_patches}")
    print(f"> NUM POSITIVE CLASS4: {num_class4_patches}")
    print(f"> NUM NEGATIVE PATCHES: {num_negative_patches}")

    num_of_archives = generate_npz_files(class1_patches, class2_patches,
                                         class4_patches, negative_patches,
                                         num_images_per_archive, full_dirname,
                                         num_of_archives)

    print_parameters(num_of_archives, num_images_per_archive,
                     num_class1_patches, num_class2_patches,
                     num_class4_patches, num_negative_patches)

    # ==== We save the metadata
    f = open(os.path.join(patches_dir, metadata_dir), "w+")
    f.write(str(num_of_archives * num_images_per_archive))
    f.close()
    print("Finished saving to npz files.")


def generate_dataset_3d_ref(path, folders_pati, patches_dir, metadata_dir,
                            subdirname, normalization, boxcox_lambda, flipping,
                            fixed_range, depth_crop, shape_with_padding,
                            mask_filtering, dataset='miccaibrats',
                            images_per_file=1024, batch_size=20):
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

    ptch_imgs, ptch_lbls, n_files = {}, {}, 0
    for i, folder in enumerate(folders_pati, 1):  # For each patient's folder
        print(f"\nPatient no. {i} of {len(folders_pati)} "
              f"(Batch: {i % batch_size}/{batch_size}).")
        imgs = {}  # Here I'll store the opened images for this patient.
        for im_type in [suffs_img, suffs_lbl]:  # For images and labels
            for suff_i in im_type:  # For each suffix
                opened_file = os.path.join(path, folder, folder + suff_i)
                print('------>', opened_file)
                imgs[suff_i] = np.asanyarray(load_nii(opened_file).get_fdata())

        # TODO Assert that the shapes are the same.

        for suff, img in imgs.items():  # For each image
            # Apply preprocessing
            if depth_crop:
                (depth_start, depth_end) = depth_crop
                print(f"Cropping depth of MRI by: [{depth_start},{depth_end}]")
                imgs[suff] = img[:, :, depth_start:depth_end]

            if normalization:
                imgs[suff] = normalization(img)

            if shape_with_padding:
                print(f"Adding padding to the first two coordinates by: "
                      f"[{shape_with_padding}]")
                imgs[suff] = add_padding(img, shape_with_padding)

            if fixed_range:
                # Define fixed_range as follows:
                # fixed_range = {'_flair.nii.gz': [min_value, max_value],
                #                'label.nii.gz': [min_value, max_value]}
                if suff in fixed_range:
                    min_rng, max_rng = fixed_range[suff]
                    print(f"Fixing range to [{min_rng}, {max_rng}]")
                    imgs[suff] = 2. * (img - min_rng) / (max_rng - min_rng) - 1

            # Apply augmentation
            # TODO Flipping was not used finally, add it?
            # if flipping:
            #     print('Data augmentation: flipping')
            #     axis_to_flip = 2
            #     imgs[suff] = np.flip(img, axis_to_flip)

            # Generate im_ptchs
            print(f"Generating im_ptchs for image {folder}")

            # For the patch extractor, if there is more than one label image,
            # the first one will be used.
            label_id = suffs_lbl[0]
            ptch_im, ptch_lb = generate_patch_4_classes_ref(imgs,
                                                            label_id=label_id)

            print("Patches per class: ")
            for p_class, images in ptch_im.items():  # Images
                print(f"class{p_class} im_ptchs obtained: {len(images)}.")

                # Add them to the list with the other subject image im_ptchs
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

            n_files = generate_npz_files_ref(ptch_imgs, ptch_lbls,
                                             images_per_file, full_dirname,
                                             dataset=dataset, n_file=n_files)

            # print_parameters(n_files, images_per_file,
            #                  num_class1_patches, num_class2_patches,
            #                  num_class4_patches, num_negative_patches)

            # ==== We save the metadata
            f = open(os.path.join(patches_dir, metadata_dir), "a+")
            f.write(str(n_files * images_per_file))
            f.close()
            print("Finished saving to npz files.")
            del ptch_imgs, ptch_lbls
            ptch_imgs, ptch_lbls = {}, {}


def generate_train_val(dataset, flipping=False, normalization=None,
                       boxcox_lambda=None, produce_hold_out=False,
                       input_train_paths=None, mask_filtering=False,
                       fixed_range=False, input_hold_out_paths=None,
                       depth_crop=None, shape_with_padding=None):
    print('Training and validation path: ', dataset.origin_directory)
    path = dataset.origin_directory

    files = dataset.image_files
    random.shuffle(files)

    n_train_val = int(min(len(files) * 0.8, dataset.n_images))
    n_val = round(n_train_val * 0.06)
    print('---------------------------------n_train_val: ', n_train_val)

    if produce_hold_out:
        # In case of producing hold out (the rest of the imgs, 15% min)
        n_hold_out = max(round(len(files) * 0.2), (len(files) - n_train_val))
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
        generate_dataset_3d_ref(path, val_paths, dataset.patches_directory,
                                "metadata_val.txt", "val", normalization,
                                boxcox_lambda, flipping, None, depth_crop,
                                shape_with_padding, mask_filtering,
                                dataset.name)
        print(f"Finished creating val dataset on {dataset.patches_directory}.")

    # ===== We generate train and val datasets
    print("Generating train dataset...")
    generate_dataset_3d_ref(path, train_paths, dataset.patches_directory,
                            "metadata_train.txt", "train", normalization,
                            boxcox_lambda, flipping, None, depth_crop,
                            shape_with_padding, mask_filtering, dataset.name)
    print(f"Finished creating train dataset on {dataset.patches_directory}.")


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


def build_dataset(orig_dir, patches_dir, dataset='miccaibrats'):
    generate_train_val(Dataset(dataset, orig_dir, patches_dir))


if __name__ == "__main__":
    nargs = len(sys.argv)

    if nargs == 2 and sys.argv[1].lower() in ['-h', '--help']:
        print(description)
        print(avail)
        sys.exit(0)

    if nargs == 1:  # Default directories (no args given, kept for compatib.)
        origin_directory = "./datasets_raw/MICCAI_BraTS2020_TrainingData"
        patches_directory = "patches_brats_marzoII/miccaibrats"
        dataset_name = 'brats'
        default_msg = 'No args provided. Continue with default values? ' \
                      f'[ORIG_DIR]: "{origin_directory}" ' \
                      f'[PATCH_DIR]: "{patches_directory}"  '
        if not click.confirm(default_msg, default=True):
            sys.exit(0)

    dataset_name = sys.argv[1].lower() if nargs >= 2 else None
    origin_directory = sys.argv[2] if nargs >= 3 else None
    patches_directory = sys.argv[3] if nargs >= 4 else None

    if not os.path.exists(origin_directory):
        raise FileNotFoundError(f"The origin directory does not exist "
                                f"({origin_directory}).")
    if not dataset_name:
        avail = ', '.join(DATASET_NAMES)
        raise AttributeError(f"You must enter a valid dataset type ({avail}).")
    if not patches_directory:
        patches_directory = os.path.join(origin_directory, 'im_ptchs')
        os.makedirs(patches_directory, exist_ok=True)
        print(f"Patches dir not set. Setting it as: {patches_directory}.")

    if dataset_name == 'brats':
        build_dataset(origin_directory, patches_directory)
    elif dataset_name == 'hepaticvessel':
        build_dataset(origin_directory, patches_directory, 'hepaticvessel')
    else:
        raise AttributeError(f"Dataset processing not implemented yet.")
