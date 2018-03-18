import cv2
import random
import numpy as np

import helper

def apply_effect(img, config):
    height, width, channels = img.shape
    coef = random.uniform(config['Blurring']['mincoef'], config['Blurring']['maxcoef'])
    sigma_x = random.uniform(config['Blurring']['minsigmax'], config['Blurring']['maxsigmax'])
    img_blurred = cv2.GaussianBlur(img, (3,3), sigmaX=sigma_x)
    blur_map = 1 - helper.generate_map(width, height, coef, config['Blurring']['maxcolor']) / 255.
    
    for channel in range(channels):
        img[:, :, channel] = blur_map[:, :] * img[:, :, channel] + (1 - blur_map[:, :]) * img_blurred[:, :, channel]

    return img