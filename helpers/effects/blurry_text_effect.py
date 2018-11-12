import cv2
import random
import numpy as np

from helpers.effects import helper

def apply_effect(img, config):
    height, width, channels = img.shape
    freq = random.uniform(config['Blurring']['minfreq'], config['Blurring']['maxfreq'])
    sigma_x = random.uniform(config['Blurring']['minsigmax'], config['Blurring']['maxsigmax'])
    max_color = config['Blurring']['maxcolor']
    img_blurred = cv2.GaussianBlur(img, (9,9), sigmaX=sigma_x)
    blur_map = 1 - helper.generate_map(width, height, freq, max_color) / 255.
    
    for channel in range(channels):
        img[:, :, channel] = blur_map[:, :] * img[:, :, channel] + (1 - blur_map[:, :]) * img_blurred[:, :, channel]

    return img