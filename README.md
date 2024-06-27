# lightweight_unet

This repository contains documented code of a lightweight U-Net model for ultrasound image segmentation.
This work is the result of the master thesis of Anders Kjelsrud, 
and you should cite his thesis if you use this code.
At the time of writing his thesis is not yet public. You can find a pdf copy in this repository:
[masters_thesis_Anders_Kjelsrud.pdf](masters_thesis_Anders_Kjelsrud.pdf)
Table 4.11 on page 70 is particularly interesting: the number of anatomical outliers for a model trained on
HUNT4 and tested on CAMUS is halved compared to the baseline U-Net 1. Section 4.11 on page 65 described the
final model.

The original GitHub repository of Anders can be found here:
[https://github.com/Anderzz/masters-thesis](https://github.com/Anderzz/masters-thesis).
This repository contains a cleaned up and documented version of the code.

The code uses PyTorch and is inspired by both U-Net 1 from
```
S. Leclerc, E. Smistad, J. Pedrosa, A. Ostvik, et al.
"Deep Learning for Segmentation using an Open Large-Scale Dataset in 2D Echocardiography"
in IEEE Transactions on Medical Imaging, vol. 38, no. 9, pp. 2198-2210, Sept. 2019.
```
and nnU-Net
```
Isensee, Fabian, et al. "nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation."
Nature methods 18.2 (2021): 203-211.
```
The result is a lightweight model that performs as well as nnU-Net on CAMUS and HUNT4,
but with an order of magnitude less parameters.


## Installation
Python 3.9 was used to develop this code.

See [requirements.txt](requirements.txt) for a list of required packages.

Alternatively, [requirements.yaml](requirements.yaml) can be used to create a conda environment.

## Inference

See [src/inference.py](src/inference.py) for an example of how to use the model for inference.

### Trained models
The following trained models are available on cius-compute:
- HUNT4: [/home/gillesv/PycharmProjects/lightweight_segmentation/src/experiments/lightweight_unet/HUNT4_a2c_a4c/lowest_val_dice.pth](/home/gillesv/PycharmProjects/lightweight_segmentation/src/experiments/lightweight_unet/HUNT4_a2c_a4c/lowest_val_dice.pth)

- CAMUS (first cv split): [/home/gillesv/PycharmProjects/lightweight_segmentation/src/experiments/lightweight_unet/camus/lowest_val_dice.pth](/home/gillesv/PycharmProjects/lightweight_segmentation/src/experiments/lightweight_unet/camus/lowest_val_dice.pth)





## TRAINING

Your data should be in the following format:
```
dataset_folder
├── numpy_files
│   ├── patient1
│   │   ├── frame1.npy
│   │   ├── frame2.npy
│   │   ├── ...
│   ├── patient2
│   ├── ...
├── splits
│   ├── test.txt
│   ├── train.txt
│   ├── val.txt
```
Where each frame.npy is a numpy array of shape (2, depth, width) where the first
channel is the image and the second channel is the ground truth segmentation mask.
The files train.txt, val.txt and test.txt contain the names of the patients that should be used for 
training, validation and testing respectively, with each line containing the name of one patient.
See 
[/home/gillesv/data/lightweight_segmentation/preprocessing_output/HUNT4_a2c_a4c_](/home/gillesv/data/lightweight_segmentation/preprocessing_output/HUNT4_a2c_a4c)
on cius-compute for an example.

You then need to specify the path to the dataset_folder in the DATA_DIR attribute in the 
config you use for training.
The default config can be found at 
[src/configs/training/default_training_config.yaml](src/configs/training/default_training_config.yaml)

Then you can run the training script:
``` bash
export PYTHONPATH=.
python src/train.py
```


## TESTING

Your data should be in the same structure as for training.

You also need to specify the path to the dataset_folder in the DATA_DIR attribute in the
config you use for testing.
The default config can be found at
[src/configs/testing/default_testing_config.yaml](src/configs/testing/default_testing_config.yaml)

You also need to specify MODEL.PATH_TO_MODEL to the trained model you want to test.

Then you can run the testing script:
``` bash
export PYTHONPATH=.
python src/test.py
```

The testing script will create a 'test_results' folder in the same directory as the model with results.




















