from __future__ import print_function
from freetype import *
import numpy as np
import cv2
import math


def _calculate_bounding_box(face, text, config):
    slot = face.glyph
    width, height, baseline = 0, 0, 0
    previous = 0
    
    for i, c in enumerate(text):
        angle = 0
        delta = 0
        try:
            angle = config["CurrentTransormations"]["rotations"][i]
            delta = config["CurrentTransormations"]["translations"][i]
        except:
            pass

        matrix = Matrix(int(math.cos(angle) * 0x10000), int(-math.sin(angle) * 0x10000),
                        int(math.sin(angle) * 0x10000), int( math.cos(angle) * 0x10000))
        
        face.set_transform(matrix, Vector(0, 0))
        face.load_char(c)
        bitmap = slot.bitmap
        height = max(height,
                     bitmap.rows + max(0,-(slot.bitmap_top-bitmap.rows)))
        baseline = max(baseline, max(0,-(slot.bitmap_top-bitmap.rows)))
        kerning = face.get_kerning(previous, c)
        width += (slot.advance.x >> 6) + (kerning.x >> 6) + delta
        previous = c

    return width * 2, height * 2, baseline


def _render_text_to_bitmap(face, text, width, height, baseline, image_array, config):
    previous_was_space = False
    previous_slot_advance_x = 0
    characters_position = []
    slot = face.glyph
    x, y = 0, 0
    previous = 0

    for i, c in enumerate(text):
        angle = 0
        delta = 0
        try:
            angle = config["CurrentTransormations"]["rotations"][i]
            delta = config["CurrentTransormations"]["translations"][i]
        except:
            pass
        
        matrix = Matrix(int(math.cos(angle) * 0x10000), int(-math.sin(angle) * 0x10000),
                        int(math.sin(angle) * 0x10000), int( math.cos(angle) * 0x10000))        
        
        face.set_transform(matrix, Vector(0, 0))
        face.load_char(c)

        if previous_was_space:
            characters_position[-1] += previous_slot_advance_x / 2
            previous_was_space = False

        bitmap = slot.bitmap
        top = slot.bitmap_top
        left = slot.bitmap_left
        w,h = bitmap.width, bitmap.rows
        y = height-baseline-top
        kerning = face.get_kerning(previous, c)

        if c == ' ':
            previous_was_space = True
            previous_slot_advance_x = (slot.advance.x >> 6)

        x += (kerning.x >> 6)
        
        image_array[y:y+h,x:x+w] += np.array(bitmap.buffer, dtype='ubyte').reshape(h, w)
        characters_position.append(x + w / 2)
        x += (slot.advance.x >> 6) + delta
        previous = c

    return characters_position


def _generate_transformations(text, config):
    result = {}

    result["rotations"] = _generate_rotations(text, config)
    result["translations"] = _generate_translations(text, config)

    return result


def _generate_rotations(text, config):
    rotations = []
    mean = config["Transformations"]["rotationmean"]
    sigma = config["Transformations"]["rotationsigma"]

    for _ in text:
        rotations.append(np.random.normal(mean, sigma))

    return rotations


def _generate_translations(text, config):
    translations = []
    mean = config["Transformations"]["translationmean"]
    sigma = config["Transformations"]["translationsignma"]

    for _ in text:
        translations.append(int(np.random.normal(mean, sigma)))

    return translations


def render_text(font, text, config):
    font_size = 0
    try:
        font_size = config["FontSizes"][font]
    except KeyError:
        font_size = calculate_font_size(font, config)
        config["FontSizes"][font] = font_size
    
    config["CurrentTransormations"] = _generate_transformations(text, config)

    face = Face(font)
    face.set_char_size(font_size)

    width, height, baseline = _calculate_bounding_box(face, text, config);

    img = np.zeros((height,width), dtype=np.ubyte)

    positions = _render_text_to_bitmap(face, text, width, height, baseline, img, config)

    annotations = zip(text, positions)

    img = _remove_trailing_space(img)
        
    return _grayscale_to_rgba(img), annotations, baseline


def calculate_font_size(font, config):
    font_size = 320
    text = "abcdefghijklmnopqrstuvwxyz"
    height_epsilon = 2

    face = Face(font)
    face.set_char_size(font_size)

    _, height, _= _calculate_bounding_box(face, text, config);
    height /= 2

    target_height = config["OutputSize"]["lineheight"]
    lower_bound = target_height - height_epsilon
    upper_bound = target_height + height_epsilon

    while not (lower_bound <= height <= upper_bound):
        if height < target_height:
            font_size += 1
        else:
            font_size -= 1
        
        face.set_char_size(font_size)
        _, height, _= _calculate_bounding_box(face, text, config);
        height /= 2

    return font_size


def _grayscale_to_rgba(img):
    w, h = img.shape
    ret = np.empty((w, h, 4), dtype=np.uint8)
    ret[:, :, 0] = 255 - img
    ret[:, :, 1] = 255 - img
    ret[:, :, 2] = 255 - img
    ret[:, :, 3] = img
    return ret


def _remove_trailing_space(img):
    while sum(img[:, -1]) == 0:
        img = np.delete(img, -1, 1)

    while sum(img[-1, :]) == 0:
        img = np.delete(img, -1, 0)
        
    while sum(img[0, :]) == 0:
        img = np.delete(img, 0, 0)

    return img
