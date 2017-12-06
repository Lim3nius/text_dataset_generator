
import random
import numpy as np

from helpers import text_renderer

def apply_effect(img, text, word_dict, config):
    img_height, img_width, _ = img.shape

    surrounding_text_left_right = ""
    while len(surrounding_text_left_right) < len(text) * 5:
        word = word_dict.keys()[random.randint(0, len(word_dict.keys()) - 1)]
        if word == text:
            continue

        surrounding_text_left_right += word + " "

    surrounding_text_img_left_right, _ = text_renderer.render_text(config['Common']['font'], surrounding_text_left_right, config['Common']['fontsize'])

    surrounding_text_top_bottom = []

    number_of_lines = 4
    line_length_coef = 20

    for line in range(number_of_lines):
        current_line = ""
        while len(current_line) < len(text) * line_length_coef:
            word = word_dict.keys()[random.randint(0, len(word_dict.keys()) - 1)]
            if word == text:
                continue

            current_line += word + " "

        surrounding_text_top_bottom.append(current_line)

    min_width = None
    lines = []
    for line in surrounding_text_top_bottom:
        line_img, _ = text_renderer.render_text(config['Common']['font'], line, config['Common']['fontsize'])
        lines.append(line_img)

        line_width = line_img.shape[1]
        if min_width is None or line_width < min_width:
            min_width = line_width

    padding = np.full((config['SurroundingText']['linespace'], min_width, 4), [255, 255, 255, 0])
    surrounding_text_img_top_bottom = np.copy(padding)
        
    for line_img in lines:
        line_img = line_img[:, :min_width]
        surrounding_text_img_top_bottom = np.concatenate((surrounding_text_img_top_bottom, line_img, np.copy(padding)))
    
    top_offset = config['Padding']['top'] - config['SurroundingText']['linespace']
    top_text_img = surrounding_text_img_top_bottom[-top_offset:, :img_width]

    bottom_offset = config['Padding']['bottom'] - config['SurroundingText']['linespace']
    bottom_text_img = surrounding_text_img_top_bottom[:bottom_offset, -img_width:]

    left_offset = config['Padding']['left'] - config['SurroundingText']['wordspace']
    left_text_img = surrounding_text_img_left_right[:, -left_offset:]

    right_offset = config['Padding']['right'] - config['SurroundingText']['wordspace']
    right_text_img = surrounding_text_img_left_right[:, :right_offset]
        
    img[:top_offset, :min(img_width, top_text_img.shape[1])] = top_text_img
    img[-bottom_offset:, :min(img_width, bottom_text_img.shape[1])] = bottom_text_img

    img_top_offset = (img_height - left_text_img.shape[0]) / 2

    img[img_top_offset:img_top_offset + left_text_img.shape[0], :left_offset] = left_text_img
    img[img_top_offset:img_top_offset + right_text_img.shape[0], -right_offset:] = right_text_img

    return img