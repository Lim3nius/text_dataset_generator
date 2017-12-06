import random
import numpy as np

from helpers import text_renderer

def generate_text_line(word, word_dict, minimal_length):
    text = ""
    while len(text) < minimal_length:
        new_word = word_dict.keys()[random.randint(0, len(word_dict.keys()) - 1)]
        if new_word != word:
            text += new_word + " "

    return text


def generate_text_image(text, font, config):
    min_width = None
    lines = []
    for line in text:
        line_img, _ = text_renderer.render_text(font, line, config['Common']['fontsize'])
        lines.append(line_img)

        line_width = line_img.shape[1]
        if min_width is None or line_width < min_width:
            min_width = line_width

    padding = np.full((config['SurroundingText']['linespace'], min_width, 4), [255, 255, 255, 0])
    img = np.copy(padding)
        
    for line_img in lines:
        line_img = line_img[:, :min_width]
        img = np.concatenate((img, line_img, np.copy(padding)))

    return np.copy(img)


def set_surroundings(img, img_top_bottom, img_left_right, config):
    img_height, img_width, _ = img.shape

    top_offset = config['Padding']['top'] - config['SurroundingText']['linespace']
    top_text_img = img_top_bottom[-top_offset:, :img_width]

    bottom_offset = config['Padding']['bottom'] - config['SurroundingText']['linespace']
    bottom_text_img = img_top_bottom[:bottom_offset, -img_width:]

    left_offset = config['Padding']['left'] - config['SurroundingText']['wordspace']
    left_text_img = img_left_right[:, -left_offset:]

    right_offset = config['Padding']['right'] - config['SurroundingText']['wordspace']
    right_text_img = img_left_right[:, :right_offset]
        
    if top_offset > 0:
        img[:top_offset, :min(img_width, top_text_img.shape[1])] = top_text_img

    if bottom_offset > 0:
        img[-bottom_offset:, :min(img_width, bottom_text_img.shape[1])] = bottom_text_img

    img_top_offset = (img_height - left_text_img.shape[0]) / 2

    if left_offset > 0:
        img[img_top_offset:img_top_offset + left_text_img.shape[0], :left_offset] = left_text_img

    if right_offset > 0:
        img[img_top_offset:img_top_offset + right_text_img.shape[0], -right_offset:] = right_text_img

    return np.copy(img)

