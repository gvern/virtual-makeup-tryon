# utils/utils.py

import cv2
import numpy as np
import logging
from typing import Any, Optional

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler()]
)

def load_image(path: str) -> Optional[np.ndarray]:
    """
    Loads an image from the specified path using OpenCV.

    :param path: Path to the image file.
    :return: Loaded image as a NumPy array, or None if loading fails.
    """
    image = cv2.imread(path)
    if image is None:
        logging.error(f"Failed to load image from {path}")
    else:
        logging.debug(f"Image loaded from {path}")
    return image

def save_image(path: str, image: np.ndarray) -> bool:
    """
    Saves an image to the specified path using OpenCV.

    :param path: Path where the image will be saved.
    :param image: Image as a NumPy array.
    :return: True if saving succeeds, False otherwise.
    """
    success = cv2.imwrite(path, image)
    if success:
        logging.debug(f"Image saved to {path}")
    else:
        logging.error(f"Failed to save image to {path}")
    return success
