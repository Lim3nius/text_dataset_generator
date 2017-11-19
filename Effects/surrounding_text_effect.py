
import random

import text_renderer

def apply_effect(img, text, word_dict, config):
    img_height, img_width, _ = img.shape

    surrounding_text = ""
    while len(surrounding_text) < len(text) * 2:
        word = word_dict.keys()[random.randint(0,len(word_dict.keys()) - 1)]
        if word == text:
            continue

        surrounding_text += word + " "

    surrounding_text_img, _ = text_renderer.render_text(config['Common']['font'], surrounding_text, config['Common']['fontsize'])
    
    top_offset = config['Padding']['top'] - config['SurroundingText']['linespace']
    top_text_img = surrounding_text_img[-top_offset:]
    _, top_text_width, _ = top_text_img.shape

    bottom_offset = config['Padding']['bottom'] - config['SurroundingText']['linespace']
    bottom_text_img = surrounding_text_img[:bottom_offset]
    _, bottom_text_width, _ = bottom_text_img.shape

    left_offset = config['Padding']['left'] - config['SurroundingText']['wordspace']
    left_text_img = surrounding_text_img[:, -left_offset:]
    left_text_height, _, _ = left_text_img.shape

    right_offset = config['Padding']['right'] - config['SurroundingText']['wordspace']
    right_text_img = surrounding_text_img[:, :right_offset]
    right_text_height, _, _ = right_text_img.shape
        
    img[:top_offset, :min(img_width, top_text_width)] = top_text_img[:, :min(img_width, top_text_width)]
    img[-bottom_offset:, :min(img_width, bottom_text_width)] = bottom_text_img[:, :min(img_width, bottom_text_width)]

    img_top_offset = (img_height - left_text_height) / 2

    img[img_top_offset:img_top_offset + left_text_height, :left_offset] = left_text_img[:, :]
    img[img_top_offset:img_top_offset + right_text_height, -right_offset:] = right_text_img[:, :]

    return img