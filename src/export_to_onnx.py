import os
import sys
import yaml
import torch
import src.network as network
import src.CONST as CONST

def export_to_onnx(model, onnx_file_path,input_shape=(1, 256, 256),
                   output_shape=(8,)):
    '''
    Export a PyTorch model to onnx format.
    The onnx model will be saved to the given onnx_file_path
    :param model: torch.nn.Module
        The PyTorch model to export
    :param onnx_file_path: str
        The path where the onnx model will be saved
    :param input_shape: tuple
        The shape of an input tensor for the model
        The batch size should not be included in the input shape
    :param output_shape:
        The shape of the output tensor for the model
        The batch size should not be included in the output shape
    '''
    # prepend dummy batch size to input_shape
    input_shape_batch = (1, *input_shape)
    dummy_input = torch.randn(input_shape_batch)
    dummy_input = dummy_input.to('cpu')
    model = model.to('cpu')
    input_names = [f"Input bsx{'x'.join(str(dim) for dim in input_shape)}"]
    output_names = [f"Output bsx{'x'.join(str(dim) for dim in output_shape)}"]
    dynamic_axes = {input_names[0]: {0: 'batch_size'}}
    # Export the model
    torch.onnx.export(model,
                      dummy_input,
                      onnx_file_path,
                      verbose=True,
                      input_names=input_names,
                      output_names=output_names,
                      dynamic_axes=dynamic_axes)

def export(config):
    """
    Export the model to onnx format at the location specified in the config file.
    :param config: dict
        The configuration dictionary. Should have the following keys:
        - 'EXPORT_LOC': str
            The location where the onnx model will be saved
        - 'WEIGHTS_LOC': str
            The location of the model weights, as a .pth file
        - 'TRAIN_CONFIG_LOC': str
            The location of the model configuration file used to train the model. This config file should contain:
            - 'MODEL': dict
                The model configuration dictionary. Should have the following keys:
                - 'INPUT_SHAPE': tuple
                    The shape of the input tensor
                - 'ACTIVATION_INTER_LAYER': str
                    The activation function to use between layers
                - 'NORMALIZE_INPUT': bool
                    Whether to normalize the input
                - 'NORMALIZE_INTER_LAYER': bool
                    Whether to normalize the output of each layer
                - 'DEEP_SUPERVISION': bool
                    Whether to use deep supervision
    """
    export_loc = config['EXPORT_LOC']
    if not os.path.exists(os.path.dirname(export_loc)):
        os.makedirs(os.path.dirname(export_loc))

    # the training config contains the model architecture needed to instantiate the model
    train_config_loc = config['TRAIN_CONFIG_LOC']
    train_config = yaml.load(open(train_config_loc), Loader=yaml.loader.SafeLoader)

    # instantiate model
    model = network.lightweight_unet(
        input_shape=train_config["MODEL"]["INPUT_SHAPE"],
        activation_inter_layer=train_config["MODEL"]["ACTIVATION_INTER_LAYER"],
        normalize_input=train_config["MODEL"]["NORMALIZE_INPUT"],
        normalize_inter_layer=train_config["MODEL"]["NORMALIZE_INTER_LAYER"],
        use_deep_supervision=train_config["MODEL"]["DEEP_SUPERVISION"],
        verbose=True,
    )

    # load trained model weights
    model.load_state_dict(torch.load(config['WEIGHTS_LOC'], map_location='cpu'))

    # export to onnx
    model = model.to('cpu')
    export_to_onnx(model, export_loc)







if __name__ == "__main__":
    # load config file if provided, otherwise use default
    if len(sys.argv) > 1:
        config_loc = sys.argv[1]
    else:
        config_loc = CONST.DEFAULT_EXPORT_CONFIG_LOC
    print('Running export with config file: ' + config_loc)
    export_config = yaml.load(open(config_loc), Loader=yaml.loader.SafeLoader)
    export(export_config)
