import nibabel as nib
import os
from tqdm import tqdm
import yaml
import numpy as np
import utils
import sys
import CONST
from src.utils import text_to_set


def preprocess(config_loc):
    """
    Preprocess the dataset according to the configuration file.
    :param config_loc: str
        The location of the configuration file.
    """
    config = yaml.load(open(config_loc), Loader=yaml.loader.SafeLoader)
    if config['DATASET_TYPE'] == 'CAMUS':
        print('Converting CAMUS data to numpy format..')
        convert_camus_dataset_to_numpy(config)
    else:
        raise NotImplementedError('TODO: only CAMUS dataset is supported for now.')


def save_splits(splits, out_loc):
    """
    Save the splits to a file
    :param splits: dict
        A dictionary with the split names as keys and the split as values, e.g.
        {'train': ['patient1', 'patient2',...], 'val': ['patient3',...], 'test': ['patient4',...]}
    """
    if not os.path.exists(out_loc):
        os.makedirs(out_loc)
    for split_name, split in splits.items():
        split_loc = os.path.join(out_loc, split_name)
        with open(split_loc+'.txt', 'w') as f:
            for patient in split:
                f.write(patient + '\n')


### CAMUS specific functions ###

def get_CAMUS_splits(split_nb, splits_loc):
    """
    Get train, validation and test splits for a given split number
    :param split_nb: split number
    :param splits_loc: location of splits. This folder should have the following structure:
                       ├─subgroups_CAMUS
                       │ ├─subGroup0_testing.txt
                       │ ├─subGroup0_training.txt
                       │ ├─subGroup0_validation.txt
                       │ ├─subGroup1_testing.txt
                       │ ├─subGroup1_training.txt
                       │ ├─subGroup1_validation.txt
                       │ ├─...
    :return: train, validation and test splits
    """
    train_loc = os.path.join(str(splits_loc), f"subGroup{split_nb}_training.txt")
    val_loc = os.path.join(str(splits_loc), f"subGroup{split_nb}_validation.txt")
    test_loc = os.path.join(str(splits_loc), f"subGroup{split_nb}_testing.txt")
    train_set = text_to_set(train_loc)
    val_set = text_to_set(val_loc)
    test_set = text_to_set(test_loc)
    # print(f"\n{val_set}\n\n{test_set}\n")
    # # print the number of samples that are the same in the validation and test set
    # print(
    #     "Number of samples that are the same in the validation and test set:",
    #     len(train_set.intersection(test_set)),
    # )
    return train_set, val_set, test_set

def convert_camus_dataset_to_numpy(config):
    """
    Convert CAMUS data to numpy format with given config file.
    This function will create a folder structure in your preprocessing_out folder under the dataset id specified
    in the config file.
    The folder structure will be:
    ├─ cv_[split_nb]
    │ ├─ numpy_files
    │ │ ├─ recording1.npy
    │ │ ├─ recording2.npy
    │ │ │ ...
    │ ├─ splits
    │ │ ├─ train.txt
    │ │ ├─ val.txt
    │ │ └─ test.txt
    Where recording1.npy, recording2.npy, ... are the numpy files containing the image and ground truth data for each
    recording in the dataset as a tuple containing (image, ground_truth), where image and ground_truth are numpy arrays.
    train.txt, val.txt, test.txt are the text files containing the patient ids for the train, validation and test sets.
    Each file contains the patient names, 1 per line, for the corresponding split
    :param config: dict
        The configuration dictionary. It should have the following keys:
        - 'CAMUS': dict
            - 'DATA_LOCATION': str,
                the location of the CAMUS data
            - 'SPLITS_LOCATION': str,
                the location of the CAMUS splits
            - 'SPLIT_NB': int,
                the split number to use
        - 'PREPROCESSING_OUT_LOC': str,
            the location to save the output,
            relative to CONST.DATA_DIR
    """
    splits_loc = os.path.join(CONST.PROJECT_ROOT, config['CAMUS']['SPLITS_LOCATION'])
    splits= get_CAMUS_splits(config['CAMUS']['SPLIT_NB'], splits_loc)
    train_set,val_set,test_set=splits
    splits_dict = {'train': train_set, 'val': val_set, 'test': test_set}
    save_splits(splits_dict, os.path.join(CONST.DATA_DIR, config['PREPROCESSING_OUT_LOC'],'splits'))
    out_loc=os.path.join(CONST.DATA_DIR,config['PREPROCESSING_OUT_LOC'],'numpy_files')
    if not os.path.exists(out_loc):
        os.makedirs(out_loc)
    for patient in tqdm(os.listdir(config['CAMUS']['DATA_LOCATION'])):
        patient_path=os.path.join(config['CAMUS']['DATA_LOCATION'],patient)
        if os.path.isdir(patient_path):
            for file in os.listdir(patient_path):
                file_path=os.path.join(patient_path,file)
                # only use ED and ES (end diastole and end systole)
                # gt stands for ground truth
                if file.endswith('.nii.gz') and ('ED' in file or 'ES' in file) and 'gt' in file:
                    file_us_img=file.replace('_gt','')
                    nii_img_us  = nib.load(os.path.join(patient_path,file_us_img))
                    us_npy = nii_img_us.get_fdata()
                    us_resized=utils.resize_image(us_npy,convert_to_png=False)
                    nii_img_gt  = nib.load(file_path)
                    gt_npy = nii_img_gt.get_fdata()
                    gt_resized=utils.resize_image(gt_npy,convert_to_png=False,annotation=True)
                    save_name=file_us_img.replace('.nii.gz','')
                    img_gt_tuple = (us_resized, gt_resized)
                    # save to trainval folder in patient subfolder
                    patient_folder = os.path.join(out_loc, patient)
                    if not os.path.exists(patient_folder):
                        os.makedirs(patient_folder)
                    np.save(os.path.join(patient_folder,save_name),img_gt_tuple)


if __name__ == '__main__':
    # load config file if provided, otherwise use default
    if len(sys.argv) > 1:
        config_loc = sys.argv[1]
    else:
        config_loc = CONST.DEFAULT_PREPROCESSING_CONFIG_LOC

    preprocess(config_loc)



