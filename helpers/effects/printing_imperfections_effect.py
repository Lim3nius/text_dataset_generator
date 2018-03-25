import sys
import random
import numpy as np

import helper
from helpers import file_helper


def apply_effect(img, config):
    height, width, _ = img.shape
    generated_map = 1 - helper.generate_map_config(width, height, config) / 255.
    generated_blobs = 1 - helper.generate_map_blobs(width, height, config) / 255.

    alpha = np.copy(img[:, :, -1]) / 255.

    alpha *= generated_map
    alpha *= generated_blobs

    img[:, :, -1] = alpha * 255

    return img
