import albumentations as A
from albumentations.pytorch import ToTensorV2





def get_augmentations(augmentation_parameters):
    """
    Get the augmentations to apply to the images
    :param augmentation_parameters: dict
        The dictionary with the augmentation parameters. It can have the following keys:
        - 'RESIZE': (int,int)
            the size to resize the images to.
            If unspecified, the images will not be resized
        - 'P_SHIFT_SCALE_ROTATE': float
            the probability of applying the shift scale rotate augmentation.
            If unspecified, the augmentation will not be applied
        - 'SHIFT_LIMIT': float
            the maximum shift to apply in the shift scale rotate augmentation.
            Must be specified if 'P_SHIFT_SCALE_ROTATE' is specified
        - 'SCALE_LIMIT': float
            the maximum scale to apply in the shift scale rotate augmentation
            Must be specified if 'P_SHIFT_SCALE_ROTATE' is specified
        - 'ROTATE_LIMIT': float
            the maximum rotation to apply in the shift scale rotate augmentation
            Must be specified if 'P_SHIFT_SCALE_ROTATE' is specified
    :return: A.Compose
        The composition of augmentations to apply to the images
    """

    list_of_augmentations = []
    if 'RESIZE' in augmentation_parameters:
        resize_dim = augmentation_parameters['RESIZE']
        list_of_augmentations.append(A.Resize(resize_dim[0],resize_dim[1]))
    if 'P_SHIFT_SCALE_ROTATE' in augmentation_parameters:
        scale_limit = augmentation_parameters['SCALE_LIMIT']
        list_of_augmentations.append(A.ShiftScaleRotate(
            shift_limit=augmentation_parameters['SHIFT_LIMIT'],
            scale_limit=(scale_limit[0],scale_limit[1]),
            rotate_limit=augmentation_parameters['ROTATE_LIMIT'],
            p=augmentation_parameters['P_SHIFT_SCALE_ROTATE'],
        ))
    list_of_augmentations.append(ToTensorV2())

    transform = A.Compose(
        list_of_augmentations
    )

    return transform


def serialize_augmentations(augmentations):
    """
    Serialize Albumentations augmentations into a serializable list of dicts.
    """
    return [aug.__repr__() for aug in augmentations]
