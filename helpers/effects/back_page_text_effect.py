
import sys
import random
import numpy as np

from helpers import image_helper
from helpers import text_renderer

def generate_back_image(width, height, text, word_dict, config):
    back_text = []

    number_of_lines = 7
    line_length_coef = 10

    for line in range(number_of_lines):
        current_line = ""
        while len(current_line) < len(text) * line_length_coef:
            word = word_dict.keys()[random.randint(0, len(word_dict.keys()) - 1)]
            if word == text:
                continue

            current_line += word + " "

        back_text.append(current_line)

    min_width = sys.maxsize
    lines = []
    for line in back_text:
        line_img, _ = text_renderer.render_text(config['Common']['font'], line, config['Common']['fontsize'])
        lines.append(line_img)

        line_width = line_img.shape[1]
        if line_width < min_width:
            min_width = line_width

    padding = np.full((20, min_width, 4), [255, 255, 255, 0])
    back_text_img = np.copy(padding)
        
    for line_img in lines:
        line_img = line_img[:, :min_width]
        back_text_img = np.concatenate((back_text_img, line_img, np.copy(padding)))
            
    back_text_img = back_text_img[:, ::-1]
    back_text_img = image_helper.get_random_part_of_texture(width, height, back_text_img)
    
    return back_text_img





