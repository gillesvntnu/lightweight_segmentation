import os
import numpy as np
import torch


class Labeled_dataset(torch.utils.data.Dataset):
    "Characterizes a dataset for labeled segmentation dataset"

    def __init__(
        self,
        patient_list,
        input_dir,
        transform=None,
        verbose=True,
        return_file_loc=False,
    ):
        """
        :param patient_list: list or set of patient ids to include in dataset
        :param input_dir: directory where data is stored. This directory should contain a folder for each patient,
                            which in turn contains one file for each recording. Each file should be a .npy file
                            containing a tuple of the ultrasound image and its segmentation mask as a numpy arrays.
        :param transform: list of transforms to apply to the data. Each transform should be a callable that takes
                            an image and mask as input and returns a dictionary containing the transformed image and mask.
        :param verbose: whether to print information about dataset
        :param return_file_loc: whether to return file location of sample when loading data
        """
        self.verbose = verbose
        self.patient_list = patient_list
        self.input_dir = input_dir
        self.init_index_to_file_dict()
        self.return_file_loc = return_file_loc
        self.transform = transform
        # list the transforms
        if self.transform is not None:
            print("Transforms:")
            for transform in self.transform:
                print(f"    {transform.__class__.__name__}")

    def __len__(self):
        "Denotes the total number of patients"
        return len(self.list_IDs)

    def __getitem__(self, custom_index=None):
        """
        'Generates one sample of data'
        :param custom_index: index of sample to load. If None, a random sample will be loaded
        :return: X,y, the sample (ultrasound image) and its label (segmentation mask) as 2D numpy arrays
        """
        if custom_index is None:
            index = self.index
        else:
            index = custom_index
        # Generate data
        if self.return_file_loc:
            X, y, file_loc = self.__data_generation(index)
        else:
            X, y = self.__data_generation(index)
        y = np.squeeze(y)

        # this is where the actual augmentation happens
        if self.transform is not None:
            transformed = self.transform(image=X, mask=y)
            X = transformed["image"]
            y = transformed["mask"]

        if self.return_file_loc:
            return X, y, file_loc
        else:
            return X, y

    def __data_generation(self, index):
        "Generates data containing batch_size samples"  # X : (n_samples, *dim)
        # Generate data
        file_loc = self.index_to_file_dict[index]
        X, y = np.load(file_loc, allow_pickle=True)
        X = X.astype(np.float32)
        y = y.astype(np.float32)
        if self.return_file_loc:
            return X, y, file_loc
        else:
            return X, y

    def init_index_to_file_dict(self):
        """
        Initializes a dictionary mapping index to file location. The index is used by the pytorch dataloader to
        load samples. The file location is used to retrieve the file location of a sample when loading data.
        The function sets the following attributes:
            - self.index_to_file_dict: dictionary mapping index to file location
            - self.list_IDs: list of indices
        """
        self.index_to_file_dict = {}
        index = 0
        for patient_id in os.listdir(self.input_dir):
            if patient_id in self.patient_list:
                patient_dir = os.path.join(self.input_dir, patient_id)
                for recording in os.listdir(patient_dir):
                    recording_path = os.path.join(patient_dir, recording)
                    self.index_to_file_dict[index] = recording_path
                    index += 1
        if self.verbose:
            print("Number of recordings in dataset:" + str(index))
        self.list_IDs = np.arange(index)

