# utils/utils.py

import cv2

def load_image(path):
    """
    Loads an image from the specified path.

    :param path: Path to the image file.
    :return: Loaded image in BGR format.
    """
    image = cv2.imread(path)
    return image

def save_image(path, image):
    """
    Saves an image to the specified path.

    :param path: Path where the image will be saved.
    :param image: Image in BGR format.
    """
    cv2.imwrite(path, image)
