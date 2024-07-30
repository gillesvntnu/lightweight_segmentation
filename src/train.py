import yaml
import sys
import CONST
from datetime import datetime
import src.augmentations
from data_loading import data_loader
import torch
import os
import utils
from torch.utils.tensorboard import SummaryWriter
import network
from tqdm import tqdm
import numpy as np
import augmentations
from src.utils import get_loss


def save_config(config, augmentations, output_dir):
    """
    Save the config file in the output directory.
    The augmentations are added to the config file before saving under the key 'TRAINING/AUGMENTATION_PARAMS'
    :param config: dict,
        configuration dictionary
    :param augmentations: dict,
        dictionary containing the serialized augmentations
    :param output_dir: str,
        directory to save the config file in
    """
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    # Define the path for the YAML file within the output directory
    config_yaml_path = os.path.join(output_dir, "config.yaml")

    try:
        # Open the file in write mode and save the configuration
        with open(config_yaml_path, "w") as file:
            # Update the configuration dictionary with the serialized augmentations
            config["TRAINING"]["AUGMENTATION_PARAMS"] = augmentations
            # Dump the updated configuration to the YAML file
            yaml.dump(config, file)
        print(f"Configuration saved successfully to {config_yaml_path}")
    except Exception as e:
        print(f"Error saving configuration to YAML: {e}")


def run_model(dataloader, optimizer, model, loss_fn, train=True, device=None, ds=False):
    """
    Run the model for one epoch on the given dataloader
    :param dataloader: torch.utils.data.DataLoader
        dataloader to run on
    :param optimizer: torch.optim.Optimizer
        optimizer to use for training
    :param model: torch.nn.Module
        model to run
    :param loss_fn: function
        loss function to use
    :param train: bool,
        whether to train or not. If False, no gradients will be computed
    :param device: torch.device
        device to run on. Used by pytorch
    :param ds: bool,
        whether to use deep supervision or not
    :return: tuple
        (avg_loss, avg_dice_score, avg_dice_score_per_class)
        where avg_loss is the average loss over the epoch as a float
        avg_dice_score is the average dice score over the epoch as a float
        avg_dice_score_per_class is the average dice score per class over the epoch as a list of floats

    """
    losses = []
    dice_scores = []  # each list contains the dice scores for each class per batch
    for i, data in enumerate(dataloader):
        # Every data instance is an input + label pair
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)
        labels_one_hot = utils.convert_to_one_hot(labels, device=device)
        # inputs = inputs.unsqueeze(1)  # add channel dimension, but ToTensorV2 does this for us

        # Zero your gradients for every batch!
        optimizer.zero_grad()
        # Make predictions for this batch
        if train:
            model.train()
        else:
            model.eval()

        if ds:
            predictions, ds1, ds2, ds3, ds4 = model.forward(inputs)
        else:
            predictions = model.forward(inputs)  # ["out"]

        # Compute the loss and its gradients
        if not train:
            with torch.no_grad():
                if ds:
                    batch_loss = loss_fn(
                        (predictions, ds1, ds2, ds3, ds4),
                        labels_one_hot,
                    )
                else:
                    batch_loss = loss_fn(predictions, labels_one_hot)
        else:
            if ds:
                batch_loss = loss_fn(
                    (predictions, ds1, ds2, ds3, ds4),
                    labels_one_hot,
                )
            else:
                batch_loss = loss_fn(predictions, labels_one_hot)
            batch_loss.backward()

        losses.append(batch_loss.item())

        # compute dice scores
        # find the class with the highest probability
        predictions = torch.argmax(predictions, dim=1)
        predictions = predictions.cpu().numpy()
        labels = labels.cpu().numpy().astype(int)
        dice_lv = utils.dice_score(predictions.squeeze(), labels.squeeze(), [1])
        dice_myo = utils.dice_score(predictions.squeeze(), labels.squeeze(), [2])
        dice_la = utils.dice_score(predictions.squeeze(), labels.squeeze(), [3])
        dices = [dice_lv, dice_myo, dice_la]
        dice_scores.append(dices)

        if train:
            # Adjust learning weights
            optimizer.step()

    # return avg loss, avg dice score and avg dice score per class
    return np.mean(losses), np.mean(dice_scores), np.mean(dice_scores, axis=0)


def train(config, verbose=True):
    """
    Train the model according to the given config file. The function will create a new subdirectory
    in the experiments directory at the location specified in the config file. In this subdirectory,
    a logs folder will be created to save the tensorboard logs. Additionally, each time the validation
    loss is lower than the previous lowest validation loss, the model will be saved in this subdirectory.
    The structure of the subdirectory will be:
    experiments/
    ├── output_dir
    │   ├── logs
    │   │   ├── events.out.tfevents....
    │   ├── config.yaml
    │   ├── model_epoch_0_dice_x.x.pth
    │   ├── model_epoch_1_dice_y.y.pth
    │   ├── ...
    │   ├── train_log.txt
    The data directory, specified in config['DATA_DIR'] should have the following structure:
    ├── data_dir
    │   ├── numpy_files
    │   │   ├── recording1.npy
    │   │   ├── recording2.npy
    │   │   │ ...
    │   ├── splits
    │   │   ├── train.txt
    │   │   ├── val.txt
    │   │   └── test.txt
    Where recording1.npy, recording2.npy, ... are the numpy files containing the image and ground truth data for each
    recording in the dataset as a tuple containing (image, ground_truth), where image and ground_truth are numpy arrays
    of size (width, height).
    train.txt, val.txt, test.txt are the text files containing the patient ids for the train, validation and test sets.
    These files contain the patient names, 1 per line, for the corresponding split.
    See /home/gillesv/data/lightweight_segmentation/preprocessing_output/cv1 for an example the desired structure
    :param config: str or dict,
        path to the config file or the configuration dictionary itself
        See the default config file at src/configs/training/default_training_config.yaml for an example
        The config file should contain the following
        - DATA_DIR: str,
            the location of the data directory
        - REL_OUT_DIR: str,
            the relative location of the output directory. Relative to CONST.EXPERIMENTS_DIR
        - MODEL: dict,
            the model configuration
            - INPUT_SHAPE: str,
                the input shape of the model as a string
            - DEEP_SUPERVISION: bool,
                whether to use deep supervision or not
        - TRAINING: dict,
            the training configuration
            - NB_EPOCHS: int,
                the number of epochs to train for
            - LR: float,
                the learning rate
            - LOSS: str,
                the loss function to use
            - DATA_LOADER_PARAMS: dict,
                the parameters for the data loader with
                - batch_size: int,
                    the batch size
                - shuffle: bool,
                    whether to shuffle the data or not
                - num_workers: int,
                    the number of workers to use for the data loader
    :param verbose: bool,
        whether to print progress or not
    """
    if isinstance(config, str):
        config_loc = config
        if verbose:
            print("Loading training config file: " + config_loc)
        # load config
        with open(config_loc, "r") as file:
            config = yaml.load(file, Loader=yaml.loader.SafeLoader)
    if verbose:
        print(f'Running training with config: {config}')


    # load splits
    splits_loc = os.path.join(config['DATA_DIR'],'splits')
    splits = utils.load_splits(splits_loc)
    train_set = splits['train']
    val_set = splits['val']


    # set up GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Running on device: {device} for {config["TRAINING"]["NB_EPOCHS"]} epochs.')

    # Create a unique directory for this run based on the current timestamp in the experiments directory
    current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir = os.path.join(
        CONST.EXPERIMENTS_DIR, config["REL_OUT_DIR"], current_time
    )
    os.makedirs(output_dir, exist_ok=True)  # Ensure the directory exists

    # setup logfile
    log_file_path = os.path.join(output_dir, "train_log.txt")
    # The dual logger allows us to print to stdout and save to a file at the same time
    sys.stdout = utils.DualLogger(log_file_path)

    # Create a separate logs directory within the unique run directory
    logs_dir = os.path.join(output_dir, "logs")
    os.makedirs(logs_dir, exist_ok=True)

    # set up model
    model = network.lightweight_unet(
        input_shape=config["MODEL"]["INPUT_SHAPE"],
        activation_inter_layer=config["MODEL"]["ACTIVATION_INTER_LAYER"],
        normalize_input=config["MODEL"]["NORMALIZE_INPUT"],
        normalize_inter_layer=config["MODEL"]["NORMALIZE_INTER_LAYER"],
        use_deep_supervision=config["MODEL"]["DEEP_SUPERVISION"],
        verbose=verbose,
    )

    model = model.to(device)

    # Set up the augmentations
    train_transform = augmentations.get_augmentations(config["AUGMENTATIONS_TRAIN"])
    # Serialize the augmentations
    train_augmentations_serialized = src.augmentations.serialize_augmentations(train_transform)

    val_transform = augmentations.get_augmentations(config["AUGMENTATIONS_VAL"])

    # save the config to the output directory
    save_config(config, train_augmentations_serialized, output_dir)

    # print the number of parameters in the model
    if verbose:
        total_nb_params = sum(p.numel() for p in model.parameters())
        print("total number of params: " + str(total_nb_params))

    # Set up the data loaders
    data_loader_params = config["TRAINING"]["DATA_LOADER_PARAMS"]
    # train
    if verbose:
        print("Loading training data...")
    numpy_files_loc = os.path.join(config["DATA_DIR"], "numpy_files")
    dataset_train = data_loader.Labeled_dataset(
        train_set,
        numpy_files_loc,
        transform=train_transform,
    )
    dataloader_train = torch.utils.data.DataLoader(dataset_train, **data_loader_params)

    # validation
    if verbose:
        print("Loading validation data...")
    dataset_validation = data_loader.Labeled_dataset(
        val_set,
        numpy_files_loc,
        transform=val_transform
    )
    dataloader_validation = torch.utils.data.DataLoader(dataset_validation, **data_loader_params)

    # set up optimizer and scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=config["TRAINING"]["LR"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, "max", factor=0.1, patience=5
    )

    # set up tensorboard writer
    writer = SummaryWriter(log_dir=os.path.join(output_dir, "logs"))

    # calculate loss on validation set before first epoch
    loss_fn = get_loss(config["TRAINING"]["LOSS"], device, verbose=verbose)
    validation_loss, validation_dice, validation_dice_per_class = run_model(
        dataloader_validation,
        optimizer,
        model,
        loss_fn,
        train=False,
        device=device,
        ds=config["MODEL"]["DEEP_SUPERVISION"],
    )
    writer.add_scalar("Loss/validation", validation_loss, 0)
    writer.add_scalar("Dice/validation", validation_dice, 0)
    print(
        f"Epoch 0 validation dice: {round(validation_dice, 3)}, "
        f"per class: {[round(x, 3) for x in validation_dice_per_class]}"
    )
    writer.flush()

    # Early stopping setup
    epochs_without_improvement = 0
    patience = config["TRAINING"]["PATIENCE"]

    # training loop
    current_best_dice = 0
    if verbose:
        print('running training..')
    for epoch in tqdm(range(config["TRAINING"]["NB_EPOCHS"])):
        # run 1 epoch of training and validation
        train_loss, train_dice, train_dice_per_class = run_model(
            dataloader_train,
            optimizer,
            model,
            loss_fn,
            train=True,
            device=device,
            ds=config["MODEL"]["DEEP_SUPERVISION"],
        )
        validation_loss, validation_dice, validation_dice_per_class = run_model(
            dataloader_validation,
            optimizer,
            model,
            loss_fn,
            train=False,
            device=device,
            ds=config["MODEL"]["DEEP_SUPERVISION"],
        )

        scheduler.step(validation_loss)

        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Dice/train", train_dice, epoch)
        writer.add_scalar("Loss/validation", validation_loss, epoch)
        writer.add_scalar("Dice/validation", validation_dice, epoch)

        # Early stopping
        if validation_dice > current_best_dice:
            current_best_dice = validation_dice
            print(f"New best model, saving..")
            torch.save(
                model.state_dict(),
                str(output_dir) + "/lowest_val_dice.pth",
            )

            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        if epochs_without_improvement > patience:
            print(
                f"Early stopping after {patience} epochs without improvement. Best validation dice: {current_best_dice}"
            )
            break

        print(
            f'Epoch {epoch + 1}/{config["TRAINING"]["NB_EPOCHS"]} train loss: {round(train_loss,3)}, '
            f'validation loss: {round(validation_loss,3)}'
        )
        print(
            f'Epoch {epoch + 1}/{config["TRAINING"]["NB_EPOCHS"]} train dice: {round(train_dice,3)}, '
            f'validation dice: {round(validation_dice,3)},  '
            f'per class: {[round(x, 3) for x in validation_dice_per_class]}'
        )
        writer.flush()

    # training done
    writer.close()
    sys.stdout.close()
    sys.stdout = sys.__stdout__  # Restore original stdout


if __name__ == "__main__":
    # load config file if provided, otherwise use default
    if len(sys.argv) > 1:
        config_loc = sys.argv[1]
    else:
        config_loc = CONST.DEFAULT_TRAINING_CONFIG_LOC

    train(config_loc)

    # alternatively to supplying a path to a config, you can run the train function with a config dictionary (example below)
    '''
    debug_config = {
        'TRAINING': {'NB_EPOCHS': 100,
                    'LOSS': 'DICE&CE_DS',
                    'OPTIMIZER': 'Adam',
                    'LR': 0.001,
                    'PATIENCE': 20,
                    'DATA_LOADER_PARAMS':
                        {'batch_size': 32, 'shuffle': True, 'num_workers': 8}},
        'AUGMENTATIONS_TRAIN': {
            'RESIZE': [256, 256],
            'SHIFT_LIMIT': 0.1,
            'SCALE_LIMIT': [-0.2, 0.1],
            'ROTATE_LIMIT': 10,
            'P_SHIFT_SCALE_ROTATE': 0.5},
        'AUGMENTATIONS_VAL': {
            'RESIZE': [256, 256]},
        'REL_OUT_DIR': 'lightweight_unet',
        'MODEL': {
            'INPUT_SHAPE': [1, 256, 256],
            'DEEP_SUPERVISION': True,
            'NORMALIZE_INPUT': True,
            'NORMALIZE_INTER_LAYER': True,
            'ACTIVATION_INTER_LAYER': 'mish'},
        'DATA_DIR': '/home/gillesv/data/lightweight_segmentation/datasets/HUNT4_a2c_a4c'
    }
    train(debug_config)
    '''
