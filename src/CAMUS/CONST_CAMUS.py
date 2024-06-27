# file to define constants
import os

this_file_path = os.path.dirname(os.path.realpath(__file__))


DEFAULT_PREPROCESSING_CAMUS_CONFIG_LOC = os.path.join(
    this_file_path, "configs/preprocessing_camus_config.yaml"
)