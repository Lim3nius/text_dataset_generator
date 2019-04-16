from __future__ import print_function
import numpy as np
import cv2
import os
import random

def place_text_on_background(text_image, background, text_alpha_coef=0.7):
    text_image_height, text_image_width, _ = text_image.shape
    
    result = np.copy(background)
    result = get_random_part_of_texture(text_image_width, text_image_height, result)

    alpha_text = text_image[:, :, 3] / 255.0
    alpha_text = alpha_text * text_alpha_coef
    alpha_back = 1.0 - alpha_text
    
    for c in range(0, 3):
        result[:text_image_height, :text_image_width, c] = (alpha_text * text_image[:text_image_height, :text_image_width, c] + 
                                                            alpha_back * result[:text_image_height, :text_image_width, c])

    result = result[:text_image_height, :text_image_width, :]
    
    return result


def get_random_part_of_texture(width, height, texture):
    texture_height, texture_width, _ = texture.shape
    
    x = random.randint(0, texture_width)
    y = random.randint(0, texture_height)

    if x + width > texture_width:
        x = texture_width - width

    if y + height > texture_height:
        y = texture_height - height

    if x < 0 or y < 0:
        raise ValueError("Background is not big enough.", "Texture shape: " + str(texture.shape), "Text image shape: " + str((height, width)))

    result = np.copy(texture[y:y+height, x:x+width])
    
    return result
    

def draw_annotations(img, annotations, baselines, a_color=[255,0,0], b_color=[0,255,0]):
    result = np.copy(img)

    for annotation in annotations:
        x, y, w, h = map(int, annotation[1])
        result[y:y+h, x] = a_color # left edge
        result[y:y+h, x+w] = a_color # right edge
        result[y, x:x+w] = a_color # top edge
        result[y+h, x:x+w] = a_color # bottom edge

    for baseline in baselines:
        x, y, w = map(int, baseline)
        if y < img.shape[0]:
            result[y, x:x+w] = b_color
        else:
            result[-1, x:x+w] = b_color

    return result

def add_padding_to_img(img, padding_top=20, padding_bottom=20, padding_left=20, padding_right=20, value=[255,255,255,0]):
    return cv2.copyMakeBorder(img, 
                              padding_top, 
                              padding_bottom, 
                              padding_left, 
                              padding_right, 
                              cv2.BORDER_CONSTANT, 
                              value=value)

def add_random_padding(back_text, config):
    default_padding = config['Page']['padding']

    left = np.random.randint(default_padding * 0.5, default_padding * 1.5)
    top = np.random.randint(default_padding * 0.5, default_padding * 1.5)
    return add_padding_to_img(back_text,
                              padding_top=top,
                              padding_bottom=2*default_padding-top,
                              padding_left=left,
                              padding_right=2*default_padding-left)

def fit_to_dimensions(img, shape):
    target_height, target_width = shape
    current_height, current_width = img.shape[:2]

    if current_height < target_height:
        padding = target_height - current_height
        img = add_padding_to_img(img, padding_bottom=padding)
    elif current_height > target_height:
        img = img[:target_height]

    if current_width < target_width:
        padding = target_width - current_width
        img = add_padding_to_img(img, padding_right=padding)
    elif current_width > target_width:
        img = img[:, :target_width]

    return img
