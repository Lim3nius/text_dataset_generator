
import numpy as np
import cv2
import os
import random

def place_text_on_background(text_image, background):
    text_image_width, text_image_height, _ = text_image.shape
    result = np.copy(background)
    result = get_random_part_of_texture(text_image_width, text_image_height, result)

    alpha_text = text_image[:, :, 3] / 255.0
    alpha_text = alpha_text * 0.7
    alpha_back = 1.0 - alpha_text
    
    for c in range(0, 3):
        result[:text_image_width, :text_image_height, c] = (alpha_text * text_image[:, :, c] + 
                                                            alpha_back * result[:text_image_width, :text_image_height, c])

    result = result[:text_image_width, :text_image_height, :]
    result = cv2.GaussianBlur(result,(3,3),0)

    return result

def get_random_part_of_texture(width, height, texture):
    texture_width, texture_height, _ = texture.shape
    x = random.randint(0, texture_width)
    y = random.randint(0, texture_height)

    if x + width > texture_width:
        x = texture_width - width

    if y + height > texture_height:
        y = texture_height - height

    return texture[x:texture_width, y:texture_height]
    

def draw_annotations(img, annotations, color=[255,0,0]):
    result = np.copy(img)

    for annotation in annotations:
        position = annotation[1]
        result[:, position] = color

    return result

def add_padding_to_img(img, padding_top=20, padding_bottom=20, padding_left=20, padding_right=20, value=[255,255,255,0]):
    return cv2.copyMakeBorder(img, 
                              padding_top, 
                              padding_bottom, 
                              padding_left, 
                              padding_right, 
                              cv2.BORDER_CONSTANT, 
                              value=value)






