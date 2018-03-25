import random
import numpy as np
import cv2
import sys
import time

from noise import snoise3

from helpers import text_renderer

def generate_text_line(current_word, word_dict, font, config, output_size_coef=2, number_of_added_words=5):
    target_width = config["OutputSize"]["width"] * output_size_coef
    text = ""

    for i in range(number_of_added_words):
        new_word = word_dict.keys()[random.randint(0, len(word_dict.keys()) - 1)]
        if new_word != current_word:
            text += new_word + " "

    line_img, _, _ = text_renderer.render_text(font, text, config)

    while line_img.shape[1] < target_width:
        for i in range(number_of_added_words):
            new_word = word_dict.keys()[random.randint(0, len(word_dict.keys()) - 1)]
            if new_word != current_word:
                text += new_word + " "

        line_img, _, _ = text_renderer.render_text(font, text, config)

    return text


def generate_text_image(text, font, config):
    min_width = None
    lines = []
    for line in text:
        line_img, _, _ = text_renderer.render_text(font, line, config)
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
        
    img_bottom_offset = config["Padding"]["bottom"] + (config["Baseline"]["text"] - config["Baseline"]["surrounding"])

    if left_offset > 0:
        img[-(img_bottom_offset + left_text_img.shape[0]):-img_bottom_offset, :left_offset] = left_text_img

    if right_offset > 0:
        img[-(img_bottom_offset + left_text_img.shape[0]):-img_bottom_offset, -right_offset:] = right_text_img

    return np.copy(img)


def generate_map_config(width, height, config):
    freq = random.uniform(config['PrintingImperfections']['minfreq'], config['PrintingImperfections']['maxfreq'])
    max_color = config['PrintingImperfections']['maxcolor']
    subtract_min = config['PrintingImperfections']['subtractmin']

    return generate_map(width, height, freq, max_color, subtract_min)


def generate_map(width, height, freq, max_color, subtract_min=True):
    seed = random.random()
    octaves = 3
    img = np.zeros((height, width))
    
    for y in range(height):
        for x in range(width):
            img[y, x] = int(((snoise3(x / freq, y / freq, seed, octaves=octaves) + 1) / 2.0) * max_color)
        
    if subtract_min:
        img = img - np.amin(img)

    return img


def generate_map_blobs(width, height, config):
    img = np.zeros((height, width))

    number_of_blobs = random.randint(0, config['PrintingImperfections']['maxblobs'])
    for blob in range(number_of_blobs):
        row = random.randint(0, height - 1)
        col = random.randint(0, width - 1)

        img[row, col] = random.randint(config['PrintingImperfections']['minblobcolor'], config['PrintingImperfections']['maxblobcolor'])
    
    kernel = np.ones((5,5), np.uint8)
    img = cv2.dilate(img, kernel, iterations=1)
    img = cv2.GaussianBlur(img, (5,5), 0)

    return img
