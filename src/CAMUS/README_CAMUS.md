

## CAMUS dataset

You must cite the following paper if you use the CAMUS dataset:

```
S. Leclerc, E. Smistad, J. Pedrosa, A. Ostvik, et al.
"Deep Learning for Segmentation using an Open Large-Scale Dataset in 2D Echocardiography"
in IEEE Transactions on Medical Imaging, vol. 38, no. 9, pp. 2198-2210, Sept. 2019.
```

You can download the CAMUS dataset from the following link:
https://humanheart-project.creatis.insa-lyon.fr/database/#collection/6373703d73e9f0047faa1bc8/folder/63fde55f73e9f004868fb7ac

### Preporcessing CAMUS

You need to specify the path to the CAMUS dataset the preprocessing configuration file:
An example config file can be found at
[src/configs/preprocessing/default_preprocessing_config.yaml](../configs/preprocessing/default_preprocessing_config.yaml).
You should set DATA_LOCATION to the path where you downloaded the CAMUS dataset in 
nii.gz format and PREPROCESSING_OUT_LOC to the path where you want to store the preprocessed data.

Then you can run the preprocessing script:
```bash
python preprocessing.py
```

The data will now be stored in the correct format expected by 
[src/train.py](../train.py) and [src/test.py](../test.py).

A model trained on CAMUS can be found at:
[src/experiments/lightweight_unet/trained/lowest_val_dice.pth](../experiments/lightweight_unet/trained/lowest_val_dice.pth)

