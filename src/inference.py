import torch
from utils import keep_largest_component
import network
import numpy as np
import skimage
import matplotlib.pyplot as plt

class LightWeightSegmentationModel(torch.nn.Module):
    """
    A wrapper class for a segmentation model that can be used for inference.
    The model should be a torch.nn.Module with a forward method that takes a batch of images as input and returns
    a batch of predictions.
    """
    def __init__(self, model, input_shape=(1, 256,256)):
        super(LightWeightSegmentationModel, self).__init__()
        self.model = model
        self.model.eval()
        self.input_shape = input_shape


    def post_process_batch_output(self,batch_output):
        """
        Post-process the output of the model
        Only keep the largest connected component for each class
        :param batch_output: torch.Tensor
            The output of the model, with shape (batch_size, num_classes, height, width)
        :return: np.ndarray
            The post-processed predictions with shape (batch_size, height, width)
        """
        # get actual predictions from output
        predictions = torch.argmax(batch_output, dim=1)
        predictions = predictions.cpu().numpy()
        # only keep the largest connected component for each class
        predictions = keep_largest_component(predictions)
        return predictions


    def run_inference_batch(self, batch_input):
        """
        Run inference on a batch of input images
        :param batch_input: torch.Tensor
            The input images with shape (batch_size, num_channels, height, width)
        :return: torch.Tensor
            The output of the model with shape (batch_size, num_classes, height, width)
        """
        output = self.model(batch_input)
        if type(output) is tuple:
            # this means the model uses deep supervision and returns a list of outputs
            # The first output is the final output
            output = output[0]
        return output


    def preprocess_batch_input(self,batch_input):
        """
        Preprocess the input batch of images
        :param batch_input: np.ndarray or torch.Tensor
            The input images with shape (batch_size, num_channels, height, width)
        :return: torch.Tensor
            The preprocessed input images with shape (batch_size, num_channels, height, width)
        """
        # check if batch_input is a numpy array
        if isinstance(batch_input, np.ndarray):
            if len(batch_input.shape) == 3:
                # Convert to batch of size 1
                batch_input = np.expand_dims(batch_input, axis=0)
            elif len(batch_input.shape) != 4:
                raise ValueError(f"Unsupported input shape: {batch_input.shape}")
            if batch_input.shape[1:] != self.input_shape:
                raise ValueError(f"Input shape should be {self.input_shape}, got {batch_input.shape[1:]}")
            # resize the images to the input shape of the model
            batch_input = torch.tensor(batch_input).float()
        elif isinstance(batch_input, torch.Tensor):
            if len(batch_input.shape) == 3:
                # Convert to batch of size 1
                batch_input = batch_input.unsqueeze(0)
            elif len(batch_input.shape) != 4:
                raise ValueError(f"Unsupported input shape: {batch_input.shape}")
            if batch_input.shape[1:] != torch.Size(self.input_shape):
                raise ValueError(f"Input shape should be {self.input_shape}, got {list(batch_input.shape[1:])}")
        return batch_input


    def predict_batch(self, batch_input):
        """
        Predict the segmentation masks for a batch of input images
        :param batch_input: np.ndarray or torch.Tensor
            The input images with shape (batch_size, num_channels, height, width)
        :return: np.ndarray
            The predicted segmentation masks with shape (batch_size, height, width)
        """
        batch_input = self.preprocess_batch_input(batch_input)
        batch_output = self.run_inference_batch(batch_input)
        return self.post_process_batch_output(batch_output)



if __name__ == "__main__":

    # model trained on HUNT4 ( on cius-compute)
    trained_model_path =\
        '/home/gillesv/PycharmProjects/lightweight_segmentation/src/experiments/lightweight_unet/HUNT4_a2c_a4c/lowest_val_dice.pth'

    # sample from HUNT4 ( on cius-compute)
    sample_path = \
        '/home/gillesv/data/lightweight_segmentation/datasets/HUNT4_a2c_a4c/numpy_files/0004/HAND800S_US-2D_109.npy'
    # The sample has form (us_image, anno) where us_image is the ultrasound image and anno is the segmentation mask

    # load trained model
    input_shape = (1,256,256)  # Input is grayscale image with shape (1, width, depth)
    model = network.lightweight_unet(input_shape)
    model.load_state_dict(torch.load(trained_model_path))
    seg_model = LightWeightSegmentationModel(model, input_shape)

    # load sample
    x,y = np.load(sample_path) # x is of shape (256,256), y is of shape (256,256)
    # add channel dimension to x
    x = np.expand_dims(x, axis=0)

    # predict the segmentation
    predictions = seg_model.predict_batch(x)

    # plot the segmentation
    import utils
    # the ultrasound image and segmentaitons are of shape (depth, width)
    utils.plot_segmentation(
        us_image=x[0],
        anno=y,
        pred=predictions[0].squeeze(),
        show=True
    )





