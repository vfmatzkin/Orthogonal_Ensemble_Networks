""" build_dataset.py Orthogonal Ensemble Networks

This script was used for processing the BraTS dataset for OEN paper.
"""

import tensorflow as tf
import tensorflow.keras as keras
import dataset
from dataset import Dataset
#from metrics import *
#from calibration_eval import metrics_eval
from utils import *
import normalization
#from experiments import*
import os
import gc 
from nibabel import load as load_nii
import math 
import sys
import random
from random import shuffle, randint
sys.settrace


def print_parameters(NUM_OF_ARCHIVES, NUM_IMAGES_PER_ARCHIVE, NUM_CLASS1_PATCHES,NUM_CLASS2_PATCHES,NUM_CLASS4_PATCHES,NUM_NEGATIVE_PATCHES):
    print("Saving to compressed npz files...")
    print("Parameters:")
    print("  > NUM_OF_ARCHIVES = {}".format(NUM_OF_ARCHIVES))
    print("  > NUM_IMAGES_PER_ARCHIVE = {}".format(NUM_IMAGES_PER_ARCHIVE))
    print("  > NUM_CLASS1_PATCHES = {}".format(NUM_CLASS1_PATCHES))
    print("  > NUM_CLASS2_PATCHES = {}".format(NUM_CLASS2_PATCHES))
    print("  > NUM_CLASS4_PATCHES = {}".format(NUM_CLASS4_PATCHES))
    print("  > NUM_NEGATIVE_PATCHES = {}".format(NUM_NEGATIVE_PATCHES))

def generate_patch_4_classes(elem):
    
    class1_patches_single_image, class2_patches_single_image, class4_patches_single_image, negative_patches_single_image = [], [], [], []
    flair = elem["flair"]
    t1 = elem["t1"]
    t2 = elem["t2"]
    t1ce = elem["t1ce"]
    labels = elem["labels"]


    assert t1.shape == flair.shape
    assert t1.shape == labels.shape
    assert t1.shape == t2.shape
    assert t1.shape == t1ce.shape
    print('------------------------------------Dentro del generate 4 classes--- Label shape: ',labels.shape)
    for k in range(16,labels.shape[2]-16,3): 
        # k is the number of image----> Agos: k no es el numero de imagenes, sino la tercera dimensión
        for i in range(16,labels.shape[0]-16,4):
            for j in range(16,labels.shape[1]-16,4):
                # (i,j) is the center

            	new_patch_labels = labels[i-16:i+16,j-16:j+16,k-16:k+16]
            	new_patch_flair = flair[i-16:i+16,j-16:j+16,k-16:k+16]
            	new_patch_t1 = t1[i-16:i+16,j-16:j+16,k-16:k+16]
            	new_patch_t2 = t2[i-16:i+16,j-16:j+16,k-16:k+16]
            	new_patch_t1ce = t1ce[i-16:i+16,j-16:j+16,k-16:k+16]

            	if labels[i][j][k]  == 1.0:
            		class1_patches_single_image.append([new_patch_t1,new_patch_t1ce,new_patch_t2, new_patch_flair,new_patch_labels])

            	elif labels[i][j][k]  == 2.0:
            		class2_patches_single_image.append([new_patch_t1,new_patch_t1ce,new_patch_t2, new_patch_flair,new_patch_labels])
            	elif labels[i][j][k]  == 4.0:
            		class4_patches_single_image.append([new_patch_t1,new_patch_t1ce,new_patch_t2, new_patch_flair,new_patch_labels])
            	else:
            		negative_patches_single_image.append([new_patch_t1,new_patch_t1ce,new_patch_t2, new_patch_flair,new_patch_labels])
    return class1_patches_single_image,class2_patches_single_image,class4_patches_single_image, negative_patches_single_image


def write_files_metadata(dataset, train_paths, val_paths, hold_out_paths):

    print('-------> Writing files to metadata:............................... ')
    os.makedirs(os.path.dirname(dataset.patches_directory+"/metadata_files.txt"), exist_ok=True)

    f = open(os.path.join(dataset.patches_directory,"metadata_files.txt"), "w")
    for (name, paths) in [("train_paths",train_paths),("val_paths",val_paths),("hold_out_paths",hold_out_paths)]:
        f.write(name+": \n")
        for elem in paths:
            f.write(elem+" ")
        f.write("\n")
    f.close()
def generate_npz_files(class1_patches,class2_patches,class4_patches,negative_patches,NUM_IMAGES_PER_ARCHIVE, full_dirname,NUM_OF_ARCHIVES=None):
    # This function generates npz files from positive and negative patches
    # We could flip the patches on the edge Y if necessary for augmentation - but gets too noisy
    if NUM_OF_ARCHIVES==None:
    	NUM_OF_ARCHIVES = 0 # total number of archives generated
    num_of_archive = NUM_OF_ARCHIVES-1 # number of each archive
    print('Estoy generando archivos npz y el numero de archivos es: ',num_of_archive)
    # ===== We generate npz files until we have no more patches
    while 1:
        num_of_archive+=1
        t1_flair_list, labels_list = [], []
        for num_image in range(NUM_IMAGES_PER_ARCHIVE):

            #--Damos un 25% de posibilidades a que el voxel central pertenezca a cada una de las 4 clases 
            rand=randint(0,100)
            if (rand<=25):
                if len(class1_patches) == 0:
                    print('Dejo de archivar porque me quede sin archivos de clase ')
                    # No more positive patches, dataset is done
                    return NUM_OF_ARCHIVES
                data = class1_patches.pop()
            elif (25<rand<=50): 
                if len(class2_patches) == 0:
                    # No more positive patches, dataset is done
                    print('Dejo de archivar porque me quede sin archivos de clase 2')
                    return NUM_OF_ARCHIVES
                data = class2_patches.pop()
            elif (50<rand<=75): 
                if len(class4_patches) == 0:
                    print('Dejo de archivar porque me quede sin archivos de clase 4')
                    # No more positive patches, dataset is done
                    return NUM_OF_ARCHIVES
                data = class4_patches.pop()
            else:
                if len(negative_patches) == 0:
                    # No more negative patches, dataset is done
                    print('Dejo de archivar porque me quede sin archivos de clase 0')
                    return NUM_OF_ARCHIVES
                data = negative_patches.pop()
            try:
                t1, t1ce, t2, flair, labels = data
                # ===== we mix t1 and flair
                new_shape_t1_flair_joined = (4,t1.shape[0],t1.shape[1],t1.shape[2])
                t1_flair_joined = np.empty(new_shape_t1_flair_joined)
                t1_flair_joined[0]= t1
                t1_flair_joined[1]= t1ce
                t1_flair_joined[2]= t2
                t1_flair_joined[3]= flair                                

                # we swap the axes (4,32,32,32) => (32,4,32,32) => (32,32,4,32) => (32,32,32,4)
                t1_flair_joined = np.swapaxes(np.swapaxes(np.swapaxes(t1_flair_joined,0,1),1,2),2,3)

                # ===== we produce one label matrix, which is the probability of WMH.
                new_shape_one_labels = (1,labels.shape[0],labels.shape[1],labels.shape[2])
                one_label_joined = np.empty(new_shape_one_labels)
                one_label_joined[0] = labels # probability of WMH

                # ===== we swap the axes (1,32,32,32) => (32,1,32,32) => (32,32,1,32) => (32,32,32,1)
                one_label_joined = np.swapaxes(np.swapaxes(np.swapaxes(one_label_joined,0,1),1,2),2,3)

                # ===== we append them
                t1_flair_list.append(t1_flair_joined)
                labels_list.append(one_label_joined)


            except:
                raise RuntimeError("Image broken {}".format(data))

        print('-------------------------------------sali del for')
        t1_flair_list = np.array(t1_flair_list, dtype=np.float32)
        labels_list = np.array(labels_list, dtype=np.float32)
        print('labels_list centers: ',labels_list[:,16,16,16,0])
        # ==== Assertion
        assert t1_flair_list.shape == (1024, 32, 32, 32, 4)
            
        # ==== Saving the file
        outfile = os.path.join(full_dirname, "archive_number_{}.npz".format(num_of_archive))
        os.makedirs(os.path.dirname(outfile), exist_ok=True)
        np.savez_compressed(outfile, t1_flair=t1_flair_list,labels=labels_list)
        print(outfile)
        NUM_OF_ARCHIVES +=1

    return NUM_OF_ARCHIVES

def generate_dataset_3D(path,files_paths, patches_directory, metadata_dir, subdirname, \
                        normalization, boxCoxLambda, flipping, fixed_range, \
                        depth_crop, shape_with_padding,mask_filtering):

    full_dirname = os.path.join(patches_directory,subdirname)
    print('full_dirname: ',full_dirname)
    if os.path.exists(full_dirname):
        raise RuntimeError('{} already exists! Delete it if you want to regenerate archives.'.format(full_dirname))

    one_element = {}
    elem_number = 0
    class1_patches = []
    class2_patches = []
    class4_patches = []
    negative_patches = []
    NUM_OF_ARCHIVES=0
    for file_path in files_paths:

        NUM_IMAGES_PER_ARCHIVE = 1024
        NUM_CLASS1_PATCHES = len(class1_patches)
        NUM_CLASS2_PATCHES = len(class2_patches)
        NUM_CLASS4_PATCHES = len(class4_patches)
        NUM_NEGATIVE_PATCHES = len(negative_patches)
	    
        print("> NUM POSITIVE CLASS1: {}".format(NUM_CLASS1_PATCHES))
        print("> NUM POSITIVE CLASS2: {}".format(NUM_CLASS2_PATCHES))
        print("> NUM POSITIVE CLASS4: {}".format(NUM_CLASS4_PATCHES))
        print("> NUM NEGATIVE PATCHES: {}".format(NUM_NEGATIVE_PATCHES))

        # ===== We generate the patches from the loaded data ----- NUEVO I
        if elem_number==30:
        	print("Generating patches...")
        	class1_patches = []
        	class2_patches = []
        	class4_patches = []
        	negative_patches = []
        	elem_number = 0
		# ===== We generate the patches from the loaded data ----- NUEVO F

        print('------>',os.path.join(path, file_path ,  file_path+'_flair.nii.gz'))
        flair_nii_gz = load_nii(os.path.join(path, file_path ,  file_path+'_flair.nii.gz')).get_data()
        t1_nii_gz = load_nii(os.path.join(path, file_path ,  file_path+'_t1.nii.gz')).get_data()
        t1ce_nii_gz = load_nii(os.path.join(path, file_path ,  file_path+'_t1ce.nii.gz')).get_data()
        t2_nii_gz = load_nii(os.path.join(path, file_path ,  file_path+'_t2.nii.gz')).get_data()
        labels_nii_gz = load_nii(os.path.join(path,file_path, file_path + '_seg.nii.gz')).get_data()

        #Como estamos en un problema multiclase le saco la parte de set2 to0

        try:
            assert t1_nii_gz.shape == flair_nii_gz.shape
            assert t1_nii_gz.shape == labels_nii_gz.shape

        except:
            st()

        if depth_crop != None:
            (depth_start,depth_end) = depth_crop
            print("Cropping depth of MRI by: [{},{}]".format(depth_start,depth_end))
            flair_nii_gz = flair_nii_gz[:,:,depth_start:depth_end]
            t1_nii_gz = t1_nii_gz[:,:,depth_start:depth_end]
            labels_nii_gz = labels_nii_gz[:,:,depth_start:depth_end]
            brainmask_nii_gz = brainmask_nii_gz[:,:,depth_start:depth_end]

       #Eliminé la clase porque no anda y directamente trabajo con el método z-score
        if (normalization != None):

            t1_nii_gz = normalization(t1_nii_gz)
            flair_nii_gz = normalization(flair_nii_gz)
            t1ce_nii_gz = normalization(t1ce_nii_gz)
            t2_nii_gz = normalization(t2_nii_gz)


        if shape_with_padding != None:
            flair_nii_gz = add_padding(flair_nii_gz,shape_with_padding)
            t1_nii_gz = add_padding(t1_nii_gz,shape_with_padding)
            labels_nii_gz = add_padding(labels_nii_gz,shape_with_padding)
            brainmask_nii_gz = add_padding(brainmask_nii_gz,shape_with_padding)
            print("Adding padding to the first two coordinates by: [{}]".format(shape_with_padding))
    
        #print("{} image | t1 =({}/{}) | flair=({}/{}))".format(file_path, np.min(t1_nii_gz),np.max(t1_nii_gz), np.min(flair_nii_gz),np.max(flair_nii_gz)))
        #print("{} image | t2 =({}/{}) | t1ce=({}/{}))".format(file_path, np.min(t2_nii_gz),np.max(t2_nii_gz), np.min(t1ce_nii_gz),np.max(t1ce_nii_gz)))

        try:
            assert t1_nii_gz.shape == flair_nii_gz.shape
            assert t1_nii_gz.shape == labels_nii_gz.shape
            assert t1_nii_gz.shape == t1ce_nii_gz.shape
            assert t1_nii_gz.shape == t2_nii_gz.shape
        except:
            st()


        if fixed_range != None:
            (t1_min_voxel_value, t1_max_voxel_value,flair_min_voxel_value,flair_max_voxel_value) = fixed_range
            print("Fixing range to [-1,1]")
            # === we modify the range to [-1,1]
            t1_nii_gz = 2.*(t1_nii_gz - t1_min_voxel_value)/(t1_max_voxel_value-t1_min_voxel_value)-1
            flair_nii_gz = 2.*(flair_nii_gz - flair_min_voxel_value)/(flair_max_voxel_value-flair_min_voxel_value)-1

        one_element["flair"] = flair_nii_gz
        one_element["t1"] = t1_nii_gz
        one_element["labels"] = labels_nii_gz
        one_element["t2"] = t2_nii_gz
        one_element["t1ce"] = t1ce_nii_gz


        # ===== We flip the images on the Y edge for data augmentation
        data_augmentation_by_flipping = flipping
        print('Data augmentation: ',data_augmentation_by_flipping)
        if data_augmentation_by_flipping:
            axis_to_flip = 2
            one_element_flip = []
            flair_nii_gz_flip = np.flip(flair_nii_gz,axis_to_flip)
            t1_nii_gz_flip = np.flip(t1_nii_gz,axis_to_flip)
            labels_nii_gz_flip = np.flip(labels_nii_gz,axis_to_flip)

            one_element_flip.append(flair_nii_gz_flip)
            one_element_flip.append(t1_nii_gz_flip)
            one_element_flip.append(labels_nii_gz_flip)

      


        # ===== We add them to the labeled data
        print("Generating patches from image N°{}".format(elem_number))

        class1_patches_single_image,class2_patches_single_image,class4_patches_single_image, negative_patches_single_image = generate_patch_4_classes(one_element)
#Agos: patches positivos son los que tienen algunas de las 3 clases como pixel central

        print("class1_patches obtained: {}  | class2_patches obtained: {}  | class4_patches obtained: {}  | Negative patches obtained: {}".format(len(class1_patches_single_image),len(class2_patches_single_image),len(class4_patches_single_image),len(negative_patches_single_image)))
        class1_patches = class1_patches + class1_patches_single_image
        class2_patches = class2_patches + class2_patches_single_image
        class4_patches = class4_patches + class4_patches_single_image
        negative_patches = negative_patches + negative_patches_single_image
        print("class1_patches total: {}  | class2_patches obtained: {}  | class4_patches obtained: {}  | Negative patches obtained: {}".format(len(class1_patches),len(class2_patches),len(class4_patches),len(negative_patches)))

        print("Finished generating patches from image N°{}".format(elem_number))
        elem_number = elem_number+1

        print("Data loaded.")

        if elem_number==30:

	    # ==== We shuffle the patches
        	print("Patches generated. Shuffling...")
        	random.shuffle(class1_patches)
        	random.shuffle(class2_patches)
        	random.shuffle(class4_patches)
        	random.shuffle(negative_patches)
        	print("Patches shuffled.")
        	
	    # ===== Parameters
        	NUM_IMAGES_PER_ARCHIVE = 1024
        	NUM_CLASS1_PATCHES = len(class1_patches)
        	NUM_CLASS2_PATCHES = len(class2_patches)
        	NUM_CLASS4_PATCHES = len(class4_patches)
        	NUM_NEGATIVE_PATCHES = len(negative_patches)
	    
        	print("> NUM POSITIVE CLASS1: {}".format(NUM_CLASS1_PATCHES))
        	print("> NUM POSITIVE CLASS2: {}".format(NUM_CLASS2_PATCHES))
        	print("> NUM POSITIVE CLASS4: {}".format(NUM_CLASS4_PATCHES))
        	print("> NUM NEGATIVE PATCHES: {}".format(NUM_NEGATIVE_PATCHES))

        	NUM_OF_ARCHIVES = generate_npz_files(class1_patches,class2_patches,class4_patches,negative_patches,NUM_IMAGES_PER_ARCHIVE,full_dirname,NUM_OF_ARCHIVES)

        	print_parameters(NUM_OF_ARCHIVES, NUM_IMAGES_PER_ARCHIVE, NUM_CLASS1_PATCHES,NUM_CLASS2_PATCHES,NUM_CLASS4_PATCHES,NUM_NEGATIVE_PATCHES)

	    # ==== We save the metadata
        	f = open(os.path.join(patches_directory,metadata_dir), "w+")
        	f.write(str(NUM_OF_ARCHIVES*NUM_IMAGES_PER_ARCHIVE))
        	f.close()
        	print("Finished saving to npz files.")








	    # ==== We shuffle the patches
    print("Ultimos parches generados......Patches generated. Shuffling...")
    print('element number.....',elem_number)
    random.shuffle(class1_patches)
    random.shuffle(class2_patches)
    random.shuffle(class4_patches)
    random.shuffle(negative_patches)
    print("Patches shuffled.")
        	
	    # ===== Parameters
    NUM_IMAGES_PER_ARCHIVE = 1024
    NUM_CLASS1_PATCHES = len(class1_patches)
    NUM_CLASS2_PATCHES = len(class2_patches)
    NUM_CLASS4_PATCHES = len(class4_patches)
    NUM_NEGATIVE_PATCHES = len(negative_patches)
	    
    print("> NUM POSITIVE CLASS1: {}".format(NUM_CLASS1_PATCHES))
    print("> NUM POSITIVE CLASS2: {}".format(NUM_CLASS2_PATCHES))
    print("> NUM POSITIVE CLASS4: {}".format(NUM_CLASS4_PATCHES))
    print("> NUM NEGATIVE PATCHES: {}".format(NUM_NEGATIVE_PATCHES))

    NUM_OF_ARCHIVES = generate_npz_files(class1_patches,class2_patches,class4_patches,negative_patches,NUM_IMAGES_PER_ARCHIVE,full_dirname,NUM_OF_ARCHIVES)

    print_parameters(NUM_OF_ARCHIVES, NUM_IMAGES_PER_ARCHIVE, NUM_CLASS1_PATCHES,NUM_CLASS2_PATCHES,NUM_CLASS4_PATCHES,NUM_NEGATIVE_PATCHES)

	    # ==== We save the metadata
    f = open(os.path.join(patches_directory,metadata_dir), "w+")
    f.write(str(NUM_OF_ARCHIVES*NUM_IMAGES_PER_ARCHIVE))
    f.close()
    print("Finished saving to npz files.")









def generate_train_val(dataset, flipping = False, normalization = None, boxCoxLambda = None, produce_hold_out = False, \
                        full_sized_hold_out = True, input_train_paths = None, mask_filtering = False, fixed_range = False, \
                        input_hold_out_paths = None, depth_crop = None, shape_with_padding = None):

    assert fixed_range != None
    assert input_train_paths == None or input_hold_out_paths == None 
    
    # ===== We set training and val folders
    print('Training and validation path:  ',dataset.origin_directory)
    path =  dataset.origin_directory
    all_image_paths = os.listdir(path)

    random.shuffle(all_image_paths)

    # ==== We calculate the number of elems
    # == n_train_val is 85% of the images, maximum n_images

    n_train_val = int(min(len(all_image_paths)*0.8, dataset.n_images))
    print('---------------------------------n_train_val: ',n_train_val)

    # == n_hold_out is the rest of the images, minimum 15% of the images
    n_hold_out = max(round(len(all_image_paths) * 0.2),(len(all_image_paths)-n_train_val))
    n_val = round(n_train_val * 0.06)

    if produce_hold_out:
            # ===== in case of producing hold out
            n_train = n_train_val - n_val
            val_paths = all_image_paths[0:n_val]
            hold_out_paths = all_image_paths[n_val:n_hold_out+n_val]
            train_paths = all_image_paths[-n_train:]
            assert (val_paths+ train_paths + hold_out_paths).sort() == (all_image_paths).sort()
    else:
            # ===== in case of not producing hold out
            n_train = n_train_val - n_val
            hold_out_paths = []
            train_paths = all_image_paths[0:n_train]
            val_paths = all_image_paths[n_train:]

    print('Writing files to metadata')
    write_files_metadata(dataset, train_paths, val_paths, hold_out_paths)

    if (n_val == 0):
        print("Not enough images to produce validation batches. Be sure to move some train batches to the validation folder. Also update the metadata files accordingly.")
    else:
        print("Generating val dataset...")
        gen = generate_dataset_3D(path,val_paths,dataset.patches_directory,"metadata_val.txt","val",\
            normalization,boxCoxLambda,flipping, None,depth_crop,shape_with_padding,\
            mask_filtering)
        print("Finished creating val dataset on {}.".format(dataset.patches_directory))

    # ===== We generate train and val datasets
    print("Generating train dataset...")
    gen = generate_dataset_3D(path,train_paths,dataset.patches_directory,"metadata_train.txt","train",\
        normalization,boxCoxLambda,flipping, None,depth_crop,shape_with_padding,\
        mask_filtering)
    print("Finished creating train dataset on {}.".format(dataset.patches_directory))


def get_prediction_labels(prediction,labels=None):
    n_samples = prediction.shape[0]
    label_arrays = []

    for sample_number in range(n_samples):
        #print('Prediction shape: ',np.shape(prediction))
        label_data = np.argmax(prediction[sample_number], axis=0) 
        label_data[label_data == 3] = 4
        label_arrays.append(np.array(label_data, dtype=np.uint8))
    return label_arrays

def get_one_hot_prediction(y_pred):

    hard_y_pred = np.copy(y_pred)

    label_data = np.argmax(y_pred, axis=4)

    for i in range(4):    
        hard_y_pred[:,:,:,:,i][label_data[:,:,:,:]==i] = 1
        hard_y_pred[:,:,:,:,i][label_data[:,:,:,:]!=i]=0

        
    return hard_y_pred




def build_dataset_miccaibrats_1fold():

	files_path = os.listdir("./datasets_raw/MICCAI_BraTS2020_TrainingData")

	#Guardamos como ints los numeros de IDS de pacientes
	files_path_int = [int(elem[-3:]) for elem in files_path]
	files_path_int.sort()
	
	generate_train_val(Dataset(name="miccaibrats", n_images=math.inf, \
			origin_directory="./datasets_raw/MICCAI_BraTS2020_TrainingData",patches_directory="patches_brats_marzoII/miccaibrats"), \
			normalization = normalization.z_scores_normalization, produce_hold_out = True)


if __name__ == "__main__":




    if True:
        build_dataset_miccaibrats_1fold()
   

