import sys
import random
import numpy as np
import cv2

from opensimplex import OpenSimplex

from helpers import file_helper


def generate_map(config):
    simplex = OpenSimplex(seed=random.randint(0, sys.maxsize))

    width = config['OutputSize']['width']
    height = config['OutputSize']['height']
    coef = random.uniform(config['PrintingImperfections']['mincoef'], config['PrintingImperfections']['maxcoef'])
    max_color = config['PrintingImperfections']['maxcolor']

    img = np.zeros((height, width))

    for y in range(height):
        for x in range(width):
            img[y, x] = int((simplex.noise2d(x * coef, y * coef) + 1) / 2.0 * max_color)
               
    if config['PrintingImperfections']['subtractmin']:
        img = img - np.amin(img)
        
    return img


def generate_map_blobs(config):    
    width = config['OutputSize']['width']
    height = config['OutputSize']['height']

    img = np.zeros((height, width))

    number_of_blobs = random.randint(0, config['PrintingImperfections']['maxblobs'])
    for blob in range(number_of_blobs):
        row = random.randint(0, height - 1)
        col = random.randint(0, width - 1)

        img[row, col] = random.randint(config['PrintingImperfections']['minblobcolor'], config['PrintingImperfections']['maxblobcolor'])
    
    kernel = np.ones((5,5), np.uint8)
    img = cv2.dilate(img, kernel, iterations=1)
    img = cv2.GaussianBlur(img, (5,5), 0)

    return img


def apply_effect(img, config):
    height, width, _ = img.shape
    generated_map = 1 - generate_map(config) / 255.
    generated_blobs = 1 - generate_map_blobs(config) / 255.
    
    alpha = np.copy(img[:, :, -1]) / 255.
    #print(np.amin(alpha))
    #print(np.amax(alpha))
    alpha *= generated_map    
    #print(np.amin(generated_map))
    #print(np.amax(generated_map))
    alpha *= generated_blobs
    #print(np.amin(generated_blobs))
    #print(np.amax(generated_blobs))
    #print(np.amin(alpha))
    #print(np.amax(alpha))

    img[:, :, -1] = alpha * 255

    return img
