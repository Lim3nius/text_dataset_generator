
import sys
import random
import numpy as np

import helper
from helpers import image_helper
from helpers import text_renderer


def generate_back_image(width, height, text, word_dict, font, config):
    back_text = []

    number_of_lines = 7

    for line in range(number_of_lines):
        current_line = helper.generate_text_line(text, word_dict, len(text) * 10)
        back_text.append(current_line)

    back_text_img = helper.generate_text_image(back_text, font, config)
            
    back_text_img = back_text_img[:, ::-1]
    back_text_img = image_helper.get_random_part_of_texture(width, height, back_text_img)
    
    return back_text_img





