
import random
import numpy as np

import helper
from helpers import text_renderer


def apply_effect(img, text, word_dict, font, config):
    img_height, img_width, _ = img.shape

    surrounding_text_left_right = helper.generate_text_line(text, word_dict, len(text) * 5)
    surrounding_text_img_left_right, _ = text_renderer.render_text(font, surrounding_text_left_right, config['Common']['fontsize'])

    surrounding_text_top_bottom = []

    number_of_lines = 4
    for line in range(number_of_lines):
        current_line = helper.generate_text_line(text, word_dict, len(text) * 10)
        surrounding_text_top_bottom.append(current_line)

    surrounding_text_img_top_bottom = helper.generate_text_image(surrounding_text_top_bottom, font, config)    
    
    img = helper.set_surroundings(img, surrounding_text_img_top_bottom, surrounding_text_img_left_right, config)

    return img