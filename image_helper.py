
import numpy as np
import cv2

def place_text_on_background(text_image, background):
    text_image_width, text_image_height, _ = text_image.shape
    result = np.copy(background)

    alpha_text = text_image[:, :, 3] / 255.0
    alpha_text = alpha_text * 0.7
    alpha_back = 1.0 - alpha_text
    
    for c in range(0, 3):
        result[:text_image_width, :text_image_height, c] = (alpha_text * text_image[:, :, c] + 
                                                            alpha_back * result[:text_image_width, :text_image_height, c])

    result = result[:text_image_width, :text_image_height, :]
    result = cv2.GaussianBlur(result,(5,5),0)

    return result

def draw_annotations(img, annotations, color=[255,0,0]):
    result = np.copy(img)

    for annotation in annotations:
        position = annotation[1]
        result[:, position] = color

    return result

