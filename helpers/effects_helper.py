
import random

from helpers import file_helper
from helpers import image_helper
from helpers.effects import surrounding_text_effect, back_page_text_effect, printing_imperfections_effect, blurry_text_effect

import numpy as np
import cv2


def apply_effects(img, font, background, config):
    img = printing_imperfections_effect.apply_effect(img, config)

    if random.random() < config['BackText']['probability']:
        back_text = back_page_text_effect.generate_back_image(font, config)

        back_text = image_helper.add_random_padding(back_text, config)
        back_text = image_helper.fit_to_dimensions(back_text, img.shape[:2])
                
        back_text = printing_imperfections_effect.apply_effect(back_text, config)
        alpha_coef = random.uniform(config['BackText']['minalpha'], config['BackText']['maxalpha'])
        background = image_helper.place_text_on_background(back_text, background, alpha_coef)
    
    img = image_helper.place_text_on_background(img, background)

    img = blurry_text_effect.apply_effect(img, config)

    return img
