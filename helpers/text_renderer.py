from __future__ import print_function
from freetype import *
import numpy as np


def _calculate_bounding_box(face, text):
    slot = face.glyph
    width, height, baseline = 0, 0, 0
    previous = 0
    for i, c in enumerate(text):
        face.load_char(c)
        bitmap = slot.bitmap
        height = max(height,
                     bitmap.rows + max(0,-(slot.bitmap_top-bitmap.rows)))
        baseline = max(baseline, max(0,-(slot.bitmap_top-bitmap.rows)))
        kerning = face.get_kerning(previous, c)
        width += (slot.advance.x >> 6) + (kerning.x >> 6)
        previous = c

    return width * 2, height * 2, baseline


def _render_text_to_bitmap(face, text, width, height, baseline, image_array):
    previous_was_space = False
    previous_slot_advance_x = 0
    characters_position = []
    slot = face.glyph
    x, y = 0, 0
    previous = 0
    for c in text:
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
        x += (slot.advance.x >> 6)
        previous = c

    return characters_position


def render_text(font, text, font_size=32):
    font_size_coeficient = 64
    
    face = Face(font)
    face.set_char_size(font_size * font_size_coeficient)

    width, height, baseline = _calculate_bounding_box(face, text);

    img = np.zeros((height,width), dtype=np.ubyte)

    positions = _render_text_to_bitmap(face, text, width, height, baseline, img)

    img = _remove_trailing_space(img)

    annotations = zip(text, positions)
    
    return _grayscale_to_rgba(img), annotations, baseline


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
