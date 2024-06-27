import numpy as np
from medpy.metric.binary import hd, hd95
import matplotlib.pyplot as plt
import os
from PIL import Image
from scipy.ndimage import label
from skimage.transform import resize
import torch
import torch.nn as nn
import sys

np.bool = np.bool_  # fix for medpy using np.bool_ instead of bool

### preprocessing utils ###

def resize_image(
    numpy_img, resize_dim=(256, 256), annotation=False, convert_to_png=True
):
    """
    Resize numpy image to given dimensions
    :param numpy_img: numpy array
        numpy image to resize
    :param resize_dim: tuple, optional
        dimensions to resize to as a tuple (height, width)
        Default is (256, 256)
    :param annotation: bool, optional
        whether the image is an annotation or not. If True, the image is converted to one-hot encoding
        before resizing, and converted back to the original size afterwards.
         This is to avoid rounding artefacts.
        Default is False
    :param convert_to_png: bool, optional
        whether to convert the resized numpy image to a png image or not
        Default is True
    :return: numpy array or PIL image
        resized numpy image or resized png image
    """
    if annotation:
        # convert labels to one-hot encoding
        # this avoids problems with resizing and rounding errors creating artefacts
        numpy_img_one_hot = np.zeros((numpy_img.shape[0], numpy_img.shape[1], 4))
        for row in range(numpy_img.shape[0]):
            for col in range(numpy_img.shape[1]):
                numpy_img_one_hot[row, col, int(numpy_img[row, col])] = 1
        # resize each channel separately
        numpy_image_resized = np.zeros((resize_dim[0], resize_dim[1], 4))
        for i in range(1, 4):
            numpy_image_resized[:, :, i] = np.round(
                resize(numpy_img_one_hot[:, :, i], resize_dim)
            )
        # recombine to one hot encoding
        numpy_image_resized = np.argmax(numpy_image_resized, axis=2).astype(np.uint8)
    else:
        numpy_image_resized = resize(numpy_img, resize_dim)
    if convert_to_png:
        img_data = Image.fromarray(numpy_image_resized)
        img_data_grayscale = img_data.convert("L")
        return img_data_grayscale
    else:
        return numpy_image_resized

#### evaluation utils ####

def keep_largest_component(segmentation):
    """
    Keep only the largest connected component for each class in the segmentation.
    :param segmentation: np.array
        the segmentation mask with shape (1, width, height)
    :return: np.array
        the segmentation mask with only the largest connected component for each class.
        The output has the same shape as the input, i.e. (1, width, height)
    """
    # Assuming segmentation is a 2D numpy array with integer class labels
    output_segmentation = np.zeros_like(segmentation)

    # Get unique class labels, ignoring the background (assuming it's labeled as 0)
    class_labels = np.unique(segmentation)[1:]  # Skip 0 if it's the background label

    for class_label in class_labels:
        # Create a binary mask for the current class
        class_mask = segmentation == class_label

        # Perform connected component labeling
        labeled_array, num_features = label(class_mask)

        # Skip if no features are found for the class
        if num_features == 0:
            continue

        # Find the largest component
        largest_component_size = 0
        largest_component_label = 0
        for i in range(1, num_features + 1):  # Component labels are 1-indexed
            component_size = np.sum(labeled_array == i)
            if component_size > largest_component_size:
                largest_component_size = component_size
                largest_component_label = i

        # Keep only the largest component for this class in the output segmentation
        output_segmentation[labeled_array == largest_component_label] = class_label

    return output_segmentation

def dice_score(seg, gt, labels=None):
    """
    Calculate dice score for given segmentation and ground truth for the given labels.
    Both seg and gt should be numpy arrays with the same shape in label (not one-hot) format.
    :param seg: numpy array
        predicted segmentation
    :param gt: numpy array
        ground truth segmentation
    :param labels: list, optional
        labels to calculate dice score for
        If None, the default labels [1, 2, 3, 4, 5, 6] are used
    :return: float
        dice score
    """
    if labels is None:
        labels = [1, 2, 3, 4, 5, 6]
    intersection = 0
    union = 0
    for k in labels:
        intersection += np.sum(seg[gt == k] == k) * 2.0
        union += np.sum(seg[seg == k] == k) + np.sum(gt[gt == k] == k)
    dice = intersection / union
    return dice


def create_visualization(
    ultrasound,
    segmentation,
    labels=None,
    colors=np.array([(1, 0, 0), (0, 0, 1), (0, 1, 0), (1, 1, 0), (0, 1, 1), (1, 0, 1), (1, 1, 1)]),
):
    """
    Create visualization of segmentation on top of ultrasound image.
    :param ultrasound: numpy array
        ultrasound image
    :param segmentation: numpy array
        segmentation mask
    :param labels: list, optional
        labels to visualize
        If None, the default labels [1, 2, 3, 4, 5, 6] are used
    :param colors: numpy array, optional
        colors to use for visualization. There must be at least as many colors as labels.
        The order of the colors should correspond to the order of the labels.
        Default is np.array([(1, 0, 0), (0, 0, 1), (0, 1, 0), (1, 1, 0), (0, 1, 1), (1, 0, 1), (1, 1, 1)])
    :return: numpy array
        visualization of segmentation on top of ultrasound image
    """
    if labels is None:
        labels = [1, 2, 3, 4, 5, 6]
    result = np.zeros((ultrasound.shape[0], ultrasound.shape[1], 3))
    for i in range(3):
        result[:, :, i] = ultrasound / 255

    if len(labels) > len(colors):
        print("not enough colors for plotting")
        raise ValueError

    if len(segmentation.shape) == 3:
        print("one hot encoded tensor not implemented")
        raise NotImplementedError
    else:
        # Segmentation
        for i, label in enumerate(labels):
            if colors[i, 0] != 0:
                result[segmentation == label, 0] = np.clip(
                    colors[i, 0] * (0.35 + result[segmentation == label, 0]), 0.0, 1.0
                )
            if colors[i, 1] != 0:
                result[segmentation == label, 1] = np.clip(
                    colors[i, 1] * (0.35 + result[segmentation == label, 1]), 0.0, 1.0
                )
            if colors[i, 2] != 0:
                result[segmentation == label, 2] = np.clip(
                    colors[i, 2] * (0.35 + result[segmentation == label, 2]), 0.0, 1.0
                )
    return (result * 255).astype(np.uint8)


def hausdorf(seg, gt, k=1):
    """
    Calculate hausdorff distance for given segmentation and ground truth for the given label.
    Both seg and gt should be numpy arrays with the same shape in label (not one-hot) format.
    :param seg: numpy array
        predicted segmentation
    :param gt: numpy array
        ground truth segmentation
    :param k: int, optional
        label to calculate hausdorff distance for
        Default is 1
    :return: float
        hausdorff distance
    """
    return hd(seg == k, gt == k)




def boxplot(metric_values, save_dir, title, ylabel, xticks, save_name, show=True, metric="dice"):
    """
    Create boxplot of given metric values and save it to save_dir. The metric can be for example dice scores or
    hausdorff distances.
    :param metric_values: list
        list of metric scores. Each list element is a three element list with dice scores for each of the
        three class, i.e. [[metric_lv,metric_myo,metric_la],[metric_lv,metric_myo,metric_la],...]
    :param save_dir: str
        directory to save plot to
    :param title: str
        title of plot
    :param ylabel: str
        y-axis label
    :param xticks: list
        x-axis labels
    :param save_name: str
        name of file to save plot to
    :param show: bool, optional
        whether to show plot or not
        default is True
    :param metric: str, optional
        metric to plot. This affects the y-axis limits:
        - 'dice': 0-1
        - 'hausdorff': 0-200
        default is 'dice'
    """
    # create boxplot of dice scores
    fig, ax = plt.subplots()
    ax.boxplot(metric_values)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xticklabels(xticks)
    if metric == "dice":
        ax.set_ylim([0, 1])
    elif metric == "hausdorff":
        ax.set_ylim([0, 200])
    fig.tight_layout()
    # save plot
    fig.savefig(os.path.join(save_dir, save_name), dpi=300)
    if show:
        plt.show()


def plot_segmentation(us_image, anno, pred, sample_name, dices=None, plot_folder=None, show=False):
    """
    Plot annotation and prediction of a single sample
    :param us_image: numpy array
        ultrasound image
    :param anno: numpy array
        annotation of segmentation masks ('ground truth')
    :param pred: numpy array
        prediction by model
    :param sample_name: str
        name of sample
    :param dices: list, optional
        dice scores of prediction compared to annotation. This is a list with dice scores for each of the
        three class, i.e. [dice_lv,dice_myo,dice_la]
        If not specified, no dice scores are shown
    :param plot_folder: str
        folder to save plot to
        If not specified, plot is not saved
    :param show: bool, optional
        whether to show plot or not
        default is False
    """
    # plot anno and pred
    fig, ax = plt.subplots(1, 2)
    # visualization paints the segmentation on top of the ultrasound image
    visual_anno = create_visualization(
        us_image,
        anno,
        labels=[1, 2, 3],
        colors=np.array([(1, 0, 0), (0, 1, 0), (0, 0, 1)]),
    )
    ax[0].imshow(visual_anno)
    visual_pred = create_visualization(
        us_image,
        pred,
        labels=[1, 2, 3],
        colors=np.array([(1, 0, 0), (0, 1, 0), (0, 0, 1)]),
    )
    ax[1].imshow(visual_pred)
    # set titles
    ax[0].set_title("Annotation")
    ax[1].set_title("Prediction")
    # set main title
    if dices is not None:
        dice_lv, dice_myo, dice_la = dices
        fig.suptitle(
            f"{sample_name[:-4]}\n"
            f" Dice LV: {np.round(dice_lv, 2)},"
            f" Dice Myo: {np.round(dice_myo, 2)},"
            f" Dice LA: {np.round(dice_la, 2)}"
        )
    # remove axis
    ax[0].axis("off")
    ax[1].axis("off")
    # remove whitespace
    fig.tight_layout()
    # save plot
    if plot_folder is not None:
        fig.savefig(os.path.join(plot_folder, sample_name))
    if show:
        plt.show()
    plt.close(fig)


def plot_worst_predictions(worst_queue, plot_folder):
    """
    Plot the worst predictions stored in the priority queue.
    :param worst_queue: queue.PriorityQueue
        the queue containing the worst predictions
    :param plot_folder: str
        the folder to save the plots
    """
    worst_folder = os.path.join(plot_folder, "worst_cases_plots")
    os.makedirs(worst_folder, exist_ok=True)
    while not worst_queue.empty():
        loss, (input_img, label, prediction, dices, idx) = worst_queue.get()
        plot_segmentation(
            input_img.squeeze(),
            label.squeeze(),
            prediction.squeeze(),
            f"worst_{idx}.png",
            dices,
            worst_folder,
        )

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


#### preprocessing utils ####

def text_to_set(text_loc):
    """
    Convert text file to set with one element per line
    :param text_loc:
        location of text file
    :return: set
        set with one element per line
    """
    with open(text_loc) as f:
        content = f.readlines()
    content = [x.strip() for x in content]
    return set(content)

#### training utils ####

class DualLogger:
    """
    A logger class that writes messages to both the terminal and a log file.

    :param filePath: str,
        The path to the log file.
    :param mode: str, optional,
        The mode in which the log file is opened (default is 'a' for append).
    """

    def __init__(self, filePath, mode="a"):
        self.terminal = sys.stdout
        self.log = open(filePath, mode)

    def write(self, message):
        """
        Writes a message to both the terminal and the log file.

        :param message: str,
            The message to be written.
        """
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        """
        Flushes the write buffers of the log file.

        Note:
        ----
        This method ensures compatibility with Python 3, where flushing the
        stream is required. It does not flush the terminal output.
        """
        self.log.flush()

    def close(self):
        """
        Closes the log file.
        """
        self.log.close()


def convert_to_one_hot(seg, nb_classes=4, device="cpu"):
    """
    Convert segmentation mask to one-hot encoding
    :param seg: torch.Tensor
        segmentation mask
    :param nb_classes: int, optional
        number of classes including background
        Default is 4
    :param device: str, optional
        device used by pytorch
        Default is 'cpu'
    :return: torch.Tensor
        one-hot encoded segmentation mask
    """
    seg_cpu = seg.cpu()
    seg_one_hot = np.zeros(
        (seg_cpu.shape[0], nb_classes, seg_cpu.shape[1], seg_cpu.shape[2])
    )
    for i in range(nb_classes):
        seg_one_hot[:, i, :, :] = seg_cpu == i
    seg_one_hot_tensor = torch.from_numpy(seg_one_hot).to(device)
    return seg_one_hot_tensor


def get_dice_loss_fn(nb_classes=3, epsilon=1e-10, include_bg=False, one_hot=True, device="cpu"):
    """
    Return dice loss function with given parameters
    :param nb_classes: int, optional
        number of classes
        Default is 3
    :param epsilon: float, optional
        epsilon parameter for numerical stability
        Default is 1e-10
    :param include_bg: bool, optional
        whether to include background in loss calculation or not
        Default is False
    :param one_hot: bool, optional
        whether the target and output are in one-hot encoding or not. If not, they are converted to one-hot
        encoding.
        Default is True
    :param device: str, optional
        device used by pytorch
        Default is 'cpu'
    :return: function
        dice loss function with parameters (target, output)
    """
    return lambda target, output: dice_loss(
        target,
        output,
        nb_classes=nb_classes,
        epsilon=epsilon,
        include_bg=include_bg,
        device=device,
        one_hot=one_hot,
    )


def get_dice_ce_loss_fn(nb_classes=3, epsilon=1e-10, include_bg=False, one_hot=True, device="cpu"):
    """
    Return dice loss function with given parameters
    :param nb_classes: int, optional
        number of classes
        Default is 3
    :param epsilon: float, optional
        epsilon parameter for numerical stability
        Default is 1e-10
    :param include_bg: bool, optional
        whether to include background in loss calculation or not
        Default is False
    :param one_hot: bool, optional
        whether the target and output are in one-hot encoding or not. If not, they are converted to one-hot
        encoding.
        Default is True
    :param device: str, optional
        device used by pytorch
        Default is 'cpu'
    :return: function
        dice loss function with parameters (target, output)
    """
    return lambda target, output: dice_ce_loss(
        target,
        output,
        nb_classes=nb_classes,
        epsilon=epsilon,
        include_bg=include_bg,
        device=device,
        one_hot=one_hot,
    )


def get_dice_deep_supervision_loss_fn(nb_classes=3, epsilon=1e-10, include_bg=False, one_hot=True, device="cpu"):
    """
    Return dice loss function with given parameters
    :param nb_classes: int, optional
        number of classes
        Default is 3
    :param epsilon: float, optional
        epsilon parameter for numerical stability
        Default is 1e-10
    :param include_bg: bool, optional
        whether to include background in loss calculation or not
        Default is False
    :param one_hot: bool, optional
        whether the target and output are in one-hot encoding or not. If not, they are converted to one-hot
        encoding.
        Default is True
    :param device: str, optional
        device used by pytorch
        Default is 'cpu'
    :return: function
        dice loss function with parameters (target, output)
    """
    return lambda output, target: dice_deep_supervision_loss(
        output,
        target,
        nb_classes=nb_classes,
        epsilon=epsilon,
        include_bg=include_bg,
        device=device,
        one_hot=one_hot,
    )

def get_dice_ce_deep_supervision_loss_fn(nb_classes=3, epsilon=1e-10, include_bg=False, one_hot=True, device="cpu"):
    """
    Return dice and cross-entropy loss function with given parameters
    :param nb_classes: int, optional
        number of classes
        Default is 3
    :param epsilon: float, optional
        epsilon parameter for numerical stability
        Default is 1e-10
    :param include_bg: bool, optional
        whether to include background in loss calculation or not
        Default is False
    :param one_hot: bool, optional
        whether the target and output are in one-hot encoding or not. If not, they are converted to one-hot
        encoding.
        Default is True
    :param device: str, optional
        device used by pytorch
        Default is 'cpu'
    :return: function
        dice loss function with parameters (target, output)
    """
    return lambda output, target: dice_ce_deep_supervision_loss(
        output,
        target,
        nb_classes=nb_classes,
        epsilon=epsilon,
        include_bg=include_bg,
        device=device,
        one_hot=one_hot,
    )


def get_weighted_dice_loss_fn(
    class_weights=None,
    nb_classes=3,
    epsilon=1e-10,
    include_bg=False,
    one_hot=True,
    device="cpu",
):
    """
    Return weighted dice loss function with given parameters
    :param class_weights: list, optional
        class weights for weighted dice loss
        Default is None
    :param nb_classes: int, optional
        number of classes
        Default is 3
    :param epsilon: float, optional
        epsilon parameter for numerical stability
        Default is 1e-10
    :param include_bg: bool, optional
        whether to include background in loss calculation or not
        Default is False
    :param one_hot: bool, optional
        whether the target and output are in one-hot encoding or not. If not, they are converted to one-hot
        encoding.
        Default is True
    :param device: str, optional
        device used by pytorch
        Default is 'cpu'
    :return: function
        weighted dice loss function with parameters (target, output)
    """
    return lambda target, output: weighted_dice_loss(
        target,
        output,
        class_weights=class_weights,
        nb_classes=nb_classes,
        epsilon=epsilon,
        include_bg=include_bg,
        device=device,
        one_hot=one_hot,
    )


def cross_entropy_loss(output, target):
    """
    Calculate cross entropy loss for given target and output.
    :param output: Torch.Tensor
        The output of the model
    :param target: Torch.Tensor
        The 'ground truth' target segmentation
    :return: Torch.Tensor
        The cross entropy loss
    """
    criterion = nn.CrossEntropyLoss()
    return criterion(output, target)


def dice_loss(
    output,
    target,
    nb_classes=3,
    epsilon=1e-10,
    include_bg=False,
    one_hot=True,
    device="cpu",
):
    """
    Calculate dice loss for given target and output.
    :param output: Torch.Tensor
        predicted segmentation
    :param target: Torch.Tensor
        target segmentation
    :param nb_classes: int, optional
        number of classes
        default is 3
    :param epsilon: float, optional
        epsilon parameter for numerical stability
        default is 1e-10
    :param include_bg: bool, optional
        whether to include background in loss calculation or not
        default is False
    :param one_hot: bool, optional
        whether the target and output are in one-hot encoding or not. If not, they are converted to one-hot
        encoding.
    :param device: str, optional
        device used by pytorch
        default is 'cpu'
    :return: Torch.Tensor
        dice loss
    """
    if not one_hot:
        target = convert_to_one_hot(target, nb_classes=nb_classes, device=device)
        output = convert_to_one_hot(output, nb_classes=nb_classes, device=device)
    smooth = 1.0
    dice = 0
    if include_bg:
        start_idx = 0
    else:
        start_idx = 1
    for obj in range(start_idx, nb_classes):
        output_obj = output[:, obj, :, :]
        target_obj = target[:, obj, :, :]
        intersection_obj = torch.sum(output_obj * target_obj)
        union_obj = torch.sum(output_obj * output_obj) + torch.sum(
            target_obj * target_obj
        )
        dice += (2.0 * intersection_obj + smooth) / (union_obj + smooth)
    dice /= nb_classes - 1
    return -torch.clamp(dice, 0.0, 1.0 - epsilon)


def dice_deep_supervision_loss(
    output,
    target,
    nb_classes=3,
    epsilon=1e-10,
    include_bg=False,
    one_hot=True,
    device="cpu",
):
    """
    Calculate dice loss for given target and output.
    :param output: list
        list of predicted segmentations at different levels of the network (deep supervision).
        In the current implenentation, the list needs to have 5 elements: final output, and 4 deep supervision
        outputs.
    :param target: Torch.Tensor
        target 'ground truth' segmentation
    :param nb_classes: int, optional
        number of classes
        default is 3
    :param epsilon: float, optional
        epsilon parameter for numerical stability
        default is 1e-10
    :param include_bg: bool, optional
        whether to include background in loss calculation or not
        default is False
    :param one_hot: bool, optional
        whether the target and output are in one-hot encoding or not. If not, they are converted to one-hot
        encoding.
        default is True
    :param device: str, optional
        device used by pytorch
        default is 'cpu'
    :return: Torch.Tensor
        deep supervision dice loss
    """
    final_out, ds1, ds2, ds3, ds4 = output
    loss_dice = (
        dice_loss(
            final_out,
            target,
            nb_classes=nb_classes,
            epsilon=epsilon,
            include_bg=include_bg,
            one_hot=one_hot,
            device=device,
        )
        + 0.3
        * dice_loss(
            ds1,
            target,
            nb_classes=nb_classes,
            epsilon=epsilon,
            include_bg=include_bg,
            one_hot=one_hot,
            device=device,
        )
        + 0.3
        * dice_loss(
            ds2,
            target,
            nb_classes=nb_classes,
            epsilon=epsilon,
            include_bg=include_bg,
            one_hot=one_hot,
            device=device,
        )
        + 0.3
        * dice_loss(
            ds3,
            target,
            nb_classes=nb_classes,
            epsilon=epsilon,
            include_bg=include_bg,
            one_hot=one_hot,
            device=device,
        )
        + 0.3
        * dice_loss(
            ds4,
            target,
            nb_classes=nb_classes,
            epsilon=epsilon,
            include_bg=include_bg,
            one_hot=one_hot,
            device=device,
        )
    )

    return loss_dice

def dice_ce_deep_supervision_loss(
    output,
    target,
    nb_classes=3,
    epsilon=1e-10,
    include_bg=False,
    one_hot=True,
    device="cpu",
):
    """
    Calculate dice&cross entropy loss for given target and output.
    :param output: list
        list of predicted segmentations at different levels of the network (deep supervision).
        In the current implenentation, the list needs to have 5 elements: final output, and 4 deep supervision
        outputs.
    :param target: Torch.Tensor
        target 'ground truth' segmentation
    :param nb_classes: int, optional
        number of classes
        default is 3
    :param epsilon: float, optional
        epsilon parameter for numerical stability
        default is 1e-10
    :param include_bg: bool, optional
        whether to include background in loss calculation or not
        default is False
    :param one_hot: bool, optional
        whether the target and output are in one-hot encoding or not. If not, they are converted to one-hot
        encoding.
        default is True
    :param device: str, optional
        device used by pytorch
        default is 'cpu'
    :return: Torch.Tensor
        deep supervision dice loss + deep supervision cross entropy loss
    """
    final_out, ds1, ds2, ds3, ds4 = output
    loss_dice = (
        dice_loss(
            final_out,
            target,
            nb_classes=nb_classes,
            epsilon=epsilon,
            include_bg=include_bg,
            one_hot=one_hot,
            device=device,
        )
        + 0.3
        * dice_loss(
            ds1,
            target,
            nb_classes=nb_classes,
            epsilon=epsilon,
            include_bg=include_bg,
            one_hot=one_hot,
            device=device,
        )
        + 0.3
        * dice_loss(
            ds2,
            target,
            nb_classes=nb_classes,
            epsilon=epsilon,
            include_bg=include_bg,
            one_hot=one_hot,
            device=device,
        )
        + 0.3
        * dice_loss(
            ds3,
            target,
            nb_classes=nb_classes,
            epsilon=epsilon,
            include_bg=include_bg,
            one_hot=one_hot,
            device=device,
        )
        + 0.3
        * dice_loss(
            ds4,
            target,
            nb_classes=nb_classes,
            epsilon=epsilon,
            include_bg=include_bg,
            one_hot=one_hot,
            device=device,
        )
    )

    cross_entropy_loss_total = (
         cross_entropy_loss(final_out, target)
         + 0.3 * cross_entropy_loss(ds1, target)
         + 0.3 * cross_entropy_loss(ds2, target)
         + 0.3 * cross_entropy_loss(ds3, target)
         + 0.3 * cross_entropy_loss(ds4, target)
    )

    combined_loss = (loss_dice + cross_entropy_loss_total) / 2
    return combined_loss

def dice_ce_loss(
    output,
    target,
    nb_classes=3,
    epsilon=1e-10,
    include_bg=False,
    one_hot=True,
    device="cpu",
):
    """
    Calculate combined dice and cross-entropy loss for given target and output.
    :param output: Torch.Tensor
        predicted segmentation
    :param target: Torch.Tensor
        target segmentation
    :param nb_classes: int, optional
        number of classes
        default is 3
    :param epsilon: float, optional
        epsilon parameter for numerical stability
        default is 1e-10
    :param include_bg: bool, optional
        whether to include background in loss calculation or not
        default is False
    :param one_hot: bool, optional
        whether the target and output are in one-hot encoding or not. If not, they are converted to one-hot
        encoding.
        default is True
    :param device: str, optional
        device used by pytorch
        default is 'cpu'
    :return: Torch.Tensor
        combined dice and cross-entropy loss
    """
    if not one_hot:
        target = convert_to_one_hot(target, nb_classes=nb_classes, device=device)
        output = convert_to_one_hot(output, nb_classes=nb_classes, device=device)

    # Dice loss calculation
    smooth = 1.0
    dice_loss = 0
    if include_bg:
        start_idx = 0
    else:
        start_idx = 1
    for obj in range(start_idx, nb_classes):
        output_obj = output[:, obj, :, :]
        target_obj = target[:, obj, :, :]
        intersection_obj = torch.sum(output_obj * target_obj)
        union_obj = torch.sum(output_obj * output_obj) + torch.sum(
            target_obj * target_obj
        )
        dice_loss += (2.0 * intersection_obj + smooth) / (union_obj + smooth)
    dice_loss /= nb_classes - 1
    dice_loss = -torch.clamp(dice_loss, 0.0, 1.0 - epsilon)

    # Cross-entropy loss calculation
    if one_hot:
        target = torch.argmax(
            target, dim=1
        )  # Convert one-hot to indices for cross-entropy
    ce_loss = nn.CrossEntropyLoss()(output, target)

    # Combining both losses
    combined_loss = (dice_loss + ce_loss) / 2

    return combined_loss


def weighted_dice_loss(
    output,
    target,
    class_weights=None,
    nb_classes=3,
    epsilon=1e-10,
    include_bg=False,
    one_hot=True,
    device="cpu",
):
    """
    Calculate weighted dice loss for given target and output.
    :param output: Torch.Tensor
        predicted segmentation
    :param target: Torch.Tensor
        target segmentation
    :param class_weights: list, optional
        a list or tensor of weights for each class
    :param nb_classes: int, optional
        number of classes
        default is 3
    :param epsilon: float, optional
        epsilon parameter for numerical stability
        default is 1e-10
    :param include_bg: bool, optional
        whether to include background in loss calculation or not
        default is False
    :param one_hot: bool, optional
        whether the target and output are in one-hot encoding or not. If not, they are converted to one-hot
        encoding.
    :param device: str, optional
        device used by pytorch
        default is 'cpu'
    :return: Torch.Tensor
        weighted dice loss
    """
    # print(f"Using class weights: {class_weights}")
    if class_weights is None:
        class_weights = [1 for _ in range(nb_classes)]

    if not one_hot:
        target = convert_to_one_hot(target, nb_classes=nb_classes, device=device)
        output = convert_to_one_hot(output, nb_classes=nb_classes, device=device)
    smooth = 1.0
    dice_loss = 0
    start_idx = 0 if include_bg else 1

    for obj in range(start_idx, nb_classes):
        output_obj = output[:, obj, :, :]
        target_obj = target[:, obj, :, :]
        intersection_obj = torch.sum(output_obj * target_obj)
        union_obj = torch.sum(output_obj) + torch.sum(target_obj)

        # Apply class weight
        class_dice = (2.0 * intersection_obj + smooth) / (union_obj + smooth)
        weighted_dice = class_weights[obj] * class_dice  # Apply class weight here

        dice_loss += weighted_dice

    # Adjust for the number of classes considered
    if include_bg:
        dice_loss /= nb_classes
    else:
        dice_loss /= nb_classes - 1

    return -torch.clamp(dice_loss, 0.0, 1.0 - epsilon)


def load_splits(splits_dir):
    """
    Load splits from a directory containing split files.
    :param splits_dir: str
        The directory containing the split files.
        It should have the following structure:
        ├─splits_dir
        │ ├─train.txt
        │ ├─val.txt
        │ ├─test.txt
    :return: dict
        A dict containing the split for each set,
        {'train': [patient1, patient2, ...], 'val': [patient1, patient2, ...], 'test': [patient1, patient2, ...]}
    """
    result={}
    for rel_file_path in os.listdir(splits_dir):
        file_path=os.path.join(splits_dir,rel_file_path)
        with open(file_path, "r") as f:
            lines = f.readlines()
        patient_list = [x.strip() for x in lines]
        result[rel_file_path[:-4]]=patient_list
    return result


def get_loss(loss_name, device='cpu', verbose=True):
    """
    Get the loss function to use for training
    :param loss_name: str,
        name of the loss function
    :param device: torch.device, optional
        device to run on. Used by pytorch
        Default is 'cpu'
    :param verbose: bool,
        whether to print info or not
    :return: function
        loss function
        The number of parameters depends on the loss function
    """
    if loss_name == "DICE":
        loss_fn = get_dice_loss_fn(
            device=device, one_hot=True, nb_classes=4, include_bg=False
        )
    elif loss_name == "DICE_WEIGHTED":
        loss_fn = get_weighted_dice_loss_fn(
            class_weights=[1, 1, 1], device=device
        )
    elif loss_name == "DICE&CE":
        loss_fn = get_dice_ce_loss_fn(
            device=device, one_hot=True, nb_classes=4, include_bg=False
        )
    elif loss_name == "DICE_DS":
        loss_fn = get_dice_deep_supervision_loss_fn(
            device=device, one_hot=True, nb_classes=4, include_bg=False
        )
    elif loss_name == "DICE&CE_DS":
        loss_fn = get_dice_ce_deep_supervision_loss_fn(
            device=device, one_hot=True, nb_classes=4, include_bg=False
        )
    else:
        raise NotImplementedError
    if verbose:
        print(f"Using loss function: {loss_name}")
    return loss_fn


## preprocessing utils ##

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
