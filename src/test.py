import torch
import yaml
import os
import sys
import CONST
import numpy as np
from torch.utils.data import DataLoader
from data_loading import data_loader
import network
import utils
from tqdm import tqdm
import queue
from src.utils import keep_largest_component
from utils import get_loss
from src.inference import LightWeightSegmentationModel


def handle_queue(worst_queue, item, maxsize=15):
    """
    Handle the updates to the priority queue of worst predictions.
    :param worst_queue: queue.PriorityQueue
        the priority queue to update
    :param item: tuple
        the item to insert into the queue (loss, data)
    :param maxsize: int, optional
        the maximum size of the queue
    """
    if worst_queue.qsize() < maxsize:
        worst_queue.put(item)
    else:
        # If the new item has a worse (higher) loss, add it to the queue
        max_item = worst_queue.get()
        if item[0] > max_item[0]:
            worst_queue.put(item)
        else:
            worst_queue.put(max_item)


def test(config_loc, verbose=True):
    if verbose:
        print("Running testing with config file: " + config_loc)

    # load config
    with open(config_loc, "r") as file:
        config = yaml.load(file, Loader=yaml.loader.SafeLoader)

    # set up GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Running on device: {device}')

    # get parent directory of the model. This is where the plots will be saved
    out_dir = "/".join(config["MODEL"]["PATH_TO_MODEL"].split("/")[:-1])
    plot_folder = os.path.join(out_dir, "plots")
    print("Saving results to: " + plot_folder)
    os.makedirs(plot_folder, exist_ok=True)

    # get test split
    splits_loc = os.path.join(config['DATA_DIR'],'splits')
    splits = utils.load_splits(splits_loc)
    test_set = splits['test']


    # set up model
    model = network.lightweight_unet(
        input_shape=config["MODEL"]["INPUT_SHAPE"],
        activation_inter_layer=config["MODEL"]["ACTIVATION_INTER_LAYER"],
        normalize_input=config["MODEL"]["NORMALIZE_INPUT"],
        normalize_inter_layer=config["MODEL"]["NORMALIZE_INTER_LAYER"],
        use_deep_supervision=config["MODEL"]["DEEP_SUPERVISION"],
        verbose=verbose,
    )


    # load model
    model.load_state_dict(torch.load(config["MODEL"]["PATH_TO_MODEL"], map_location=device))
    model = model.to(device)

    # print number of parameters
    if verbose:
        total_nb_params = sum(p.numel() for p in model.parameters())
        print("total number of params: " + str(total_nb_params))

    # set up model for inference
    seg_model = LightWeightSegmentationModel(model, config["MODEL"]["INPUT_SHAPE"])


    # set up data loader
    data_loader_params = config["TESTING"]["DATA_LOADER_PARAMS"]
    if verbose:
        print("Loading testing data...")
    numpy_files_loc = os.path.join(config["DATA_DIR"], "numpy_files")
    dataset_test = data_loader.Labeled_dataset(
        test_set,
        numpy_files_loc,
        transform=None,
        return_file_loc=True,
    )
    dataloader_test = torch.utils.data.DataLoader(dataset_test, **data_loader_params)

    # set up loss function
    loss_fn = get_loss(config["TESTING"]["LOSS"])

    # priority queue to store the worst predictions
    worst_queue = queue.PriorityQueue(maxsize=15)

    losses = []
    dice_scores = []
    hausdorf_scores = []

    if verbose:
        print(f"Running inference on test set...")
    with torch.no_grad():
        for i, (inputs, labels, file_loc) in tqdm(
            enumerate(dataloader_test), total=len(dataloader_test)
        ):
            inputs = inputs.to(device)
            labels = labels.to(device)
            file_name = os.path.basename(file_loc[0])

            # The labels are in the format (width, height) where the values are the class labels (0, 1, 2, 3)
            # We need to convert them to one-hot in the format (width, height, num_classes)
            labels_one_hot = utils.convert_to_one_hot(labels, device=device)

            preprocessed_inputs = seg_model.preprocess_batch_input(inputs)
            predictions = seg_model.run_inference_batch(preprocessed_inputs)

            loss = loss_fn(predictions, labels_one_hot).cpu().numpy()

            labels = labels.cpu().numpy().astype(int)

            # post-processing
            predictions = seg_model.post_process_batch_output(predictions)

            # dice scores
            dice_lv = utils.dice_score(predictions.squeeze(), labels.squeeze(), [1])
            dice_myo = utils.dice_score(predictions.squeeze(), labels.squeeze(), [2])
            dice_la = utils.dice_score(predictions.squeeze(), labels.squeeze(), [3])
            dices = [dice_lv, dice_myo, dice_la]

            # hausdorf distances
            try:
                hausdorf_lv = utils.hausdorf(predictions.squeeze(), labels.squeeze(), 1)
                hausdorf_myo = utils.hausdorf(
                    predictions.squeeze(), labels.squeeze(), 2
                )
                hausdorf_la = utils.hausdorf(predictions.squeeze(), labels.squeeze(), 3)
                hausdorf_scores.append([hausdorf_lv, hausdorf_myo, hausdorf_la])
            except:
                print(f"Error calculating hausdorf distance for {file_name}")

            # plot segmentation
            utils.plot_segmentation(
                us_image=inputs[0].cpu().numpy().squeeze().T,
                anno=labels[0].squeeze().T,
                pred=predictions[0].squeeze().T,
                sample_name=f"{file_name}.png",
                dices=dices,
                plot_folder=plot_folder,
            )

            # add losses to list
            losses.append(loss)
            dice_scores.append(dices)

            # collect the worst cases
            handle_queue(
                worst_queue,
                (loss, (inputs[0].cpu().numpy(), labels[0], predictions[0], dices, i)),
            )

    # save the dice scores and hausdorf distances
    dice_scores = np.array(dice_scores)
    dice_per_class = dice_scores.T
    hausdorf_scores = np.array(hausdorf_scores)
    hausdorf_per_class = hausdorf_scores.T


    # boxplots of dice scores and hausdorf distances
    title_dice_scores = (
        f"Average Dice scores:\n LV: {np.round(np.mean(dice_per_class[0]), 2)}, "
        f"Myo: {np.round(np.mean(dice_per_class[1]), 2)}, "
        f"LA: {np.round(np.mean(dice_per_class[2]), 2)}\n"
    )

    title_hausdorf_scores = (
        f"Average Hausdorff distances:\n LV: {np.round(np.mean(hausdorf_per_class[0]), 2)}, "
        f"MYO: {np.round(np.mean(hausdorf_per_class[1]), 2)}, "
        f"LA: {np.round(np.mean(hausdorf_per_class[2]), 2)}\n"
    )

    utils.boxplot(
        hausdorf_scores,
        out_dir,
        title_hausdorf_scores,
        "Hausdorff distance",
        ["LV", "Myo", "LA"],
        "boxplot_hausdorff.png",
        metric="hausdorff",
    )
    utils.boxplot(
        dice_scores,
        out_dir,
        title_dice_scores,
        "Dice score",
        ["LV", "Myo", "LA"],
        "boxplot_dices.png",
        metric="dice",
    )

    # save the raw scores to a txt file
    raw_scores_file = os.path.join(out_dir, "raw_scores.txt")
    with open(raw_scores_file, "w") as f:
        f.write("Dice Scores:\n")
        for dice in dice_scores:
            f.write(", ".join(map(str, dice)) + "\n")

        f.write("\nHausdorff Distances:\n")
        for hausdorf in hausdorf_scores:
            f.write(", ".join(map(str, hausdorf)) + "\n")

    # save the worst predictions
    utils.plot_worst_predictions(worst_queue, plot_folder)

    # calculate average loss and dice scores
    avg_loss = np.mean(losses)
    avg_dice = np.mean(dice_scores)  # avg across all classes
    avg_dice_per_class = np.mean(dice_scores, axis=0)  # avg across all samples, but one value per class

    return avg_loss, avg_dice, avg_dice_per_class



if __name__ == "__main__":
    # load config file if provided, otherwise use default
    if len(sys.argv) > 1:
        config_loc = sys.argv[1]
    else:
        config_loc = CONST.DEFAULT_TESTING_CONFIG_LOC
    loss, dice, dice_per_class = test(config_loc)
    print(f"Average loss: {round(loss, 3)}")
    print(f"Average dice score: {round(dice, 3)}")
    print(f"Dice scores: [LV, MYO, LA] = {[round(x, 3) for x in dice_per_class]}")
