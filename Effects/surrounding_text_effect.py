
import random

import text_renderer

def apply_effect(img, text, word_dict, config):
    img_height, img_width, _ = img.shape

    surrounding_text = ""
    while len(surrounding_text) < len(text) * 3:
        word = word_dict.keys()[random.randint(0, len(word_dict.keys()) - 1)]
        if word == text:
            continue

        surrounding_text += word + " "

    surrounding_text_img, _ = text_renderer.render_text(config['Common']['font'], surrounding_text, config['Common']['fontsize'])
    
    top_offset = config['Padding']['top'] - config['SurroundingText']['linespace']
    top_text_img = surrounding_text_img[-top_offset:, :img_width]

    bottom_offset = config['Padding']['bottom'] - config['SurroundingText']['linespace']
    bottom_text_img = surrounding_text_img[:bottom_offset, -img_width:]

    left_offset = config['Padding']['left'] - config['SurroundingText']['wordspace']
    left_text_img = surrounding_text_img[:, -left_offset:]

    right_offset = config['Padding']['right'] - config['SurroundingText']['wordspace']
    right_text_img = surrounding_text_img[:, :right_offset]
        
    img[:top_offset, :min(img_width, top_text_img.shape[1])] = top_text_img
    img[-bottom_offset:, :min(img_width, bottom_text_img.shape[1])] = bottom_text_img

    img_top_offset = (img_height - left_text_img.shape[0]) / 2

    img[img_top_offset:img_top_offset + left_text_img.shape[0], :left_offset] = left_text_img
    img[img_top_offset:img_top_offset + right_text_img.shape[0], -right_offset:] = right_text_img

    return img