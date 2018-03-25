from __future__ import print_function
from freetype import *
import numpy as np
import cv2
import math
import sys

import file_helper, image_helper


def _recalculate_spaces(width, target_width, number_of_spaces, actual_space):
    diff = target_width - width
    space = diff / number_of_spaces
    spaces = [space + actual_space] * number_of_spaces
    diff -= space * number_of_spaces

    for i in range(diff):
        spaces[i] += 1

    return spaces


def _get_words(text, number_of_words):
    words = text.split()
    return " ".join(words[:number_of_words])


def _get_next_line(face, text, config):
    target_width = config["Page"]["width"]
   
    number_of_words = 1
    line = _get_words(text, number_of_words)
    width = _calculate_bounding_box(face, line, config)[0]
    
    while width < target_width and text.count(" ") >= number_of_words:
        number_of_words += 1
        line = _get_words(text, number_of_words)
        width = _calculate_bounding_box(face, line, config)[0]

    if width > target_width:
        number_of_words -= 1
        line = _get_words(text, number_of_words)

    number_of_spaces = number_of_words - 1
    width = _calculate_bounding_box(face, line, config)[0]
    
    if number_of_spaces > 0 and number_of_spaces != text.count(" "):
        config["CurrentTransformations"]["spaces"] = _recalculate_spaces(width, target_width, number_of_spaces, config["Page"]["minwordspace"])
        
    return line


def _calculate_bounding_box(face, text, config):
    slot = face.glyph
    width, height, baseline = 0, 0, 0
    previous = 0
    space_counter = 0
    
    for i, c in enumerate(text):
        if c == " ":
            space_width = config["Page"]["minwordspace"]

            try:
                space_width = config["CurrentTransformations"]["spaces"][space_counter]
            except:
                pass

            width += space_width
            space_counter += 1
            previous = 0

        else : 
            angle = 0
            delta = 0
            try:
                angle = config["CurrentTransformations"]["rotations"][i]
                delta = config["CurrentTransformations"]["translations"][i]
            except:
                pass

            matrix = Matrix(int(math.cos(angle) * 0x10000), int(-math.sin(angle) * 0x10000),
                            int(math.sin(angle) * 0x10000), int( math.cos(angle) * 0x10000))
            
            face.set_transform(matrix, Vector(0, 0))
            face.load_char(c)
            bitmap = slot.bitmap
            y = height - baseline - slot.bitmap_top
            height = max(height, bitmap.rows + max(0,-(slot.bitmap_top-bitmap.rows)))
            baseline = max(baseline, max(0,-(slot.bitmap_top-bitmap.rows)))
            kerning = face.get_kerning(previous, c)
            width += (slot.advance.x >> 6) + (kerning.x >> 6) + delta
            previous = c

    return width, height, baseline


def _render_text_to_bitmap(face, text, width, height, baseline, image_array, config):
    characters_position = []
    slot = face.glyph
    x, y = 0, 0
    previous = 0
    space_counter = 0

    for i, c in enumerate(text):
        if c == " ":
            space_width = config["Page"]["minwordspace"]

            try:
                space_width = config["CurrentTransformations"]["spaces"][space_counter]
            except:
                pass
            
            characters_position.append((x, 0, space_width, image_array.shape[0] - 1))
            x += space_width
            space_counter += 1
            previous = 0

        else:
            angle = 0
            delta = 0
            try:
                angle = config["CurrentTransformations"]["rotations"][i]
                delta = config["CurrentTransformations"]["translations"][i]
            except:
                pass
                        
            matrix = Matrix(int(math.cos(angle) * 0x10000), int(-math.sin(angle) * 0x10000),
                            int(math.sin(angle) * 0x10000), int( math.cos(angle) * 0x10000))        
            
            face.set_transform(matrix, Vector(0, 0))
            face.load_char(c)
            bitmap = slot.bitmap
            top = slot.bitmap_top
            left = slot.bitmap_left
            w, h = bitmap.width, bitmap.rows
            y = height-baseline-top
            kerning = face.get_kerning(previous, c)
            x += (kerning.x >> 6)
            image_array[y:y+h,x:x+w] += np.array(bitmap.buffer, dtype='ubyte').reshape(h, w)
            characters_position.append((x, y, w, h))
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

    for i, c in enumerate(text):
        if i < len(text) - 1:
            if text[i + 1] == " ":
                translations.append(0)
            else:
                translations.append(int(np.random.normal(mean, sigma)))

    return translations


def _update_positions(positions, img_shape, removed):
    new_positions = []
    img_height, img_width = img_shape
    t, b, l, r = removed

    for position in positions:
        left, top, width, height = position
        new_left = max(0, left - l)
        new_top = max(0, top - t)
        
        new_width = width
        if new_left + new_width >= img_width:
            new_width = img_width - new_left - 1

        new_height = height
        if new_top + new_height >= img_height:
            new_height = img_height - new_top - 1

        new_positions.append((new_left, new_top, new_width, new_height))

    return new_positions


def _annotations_increase_vertical_position(annotations, delta):
    new_annotations = []
    for annotation in annotations:
        character, position = annotation
        x, y, w, h = position
        new_annotation = (character, (x, y+delta, w, h))
        new_annotations.append(new_annotation)

    return new_annotations


def _baseline_increase_vertical_position(baselines, delta):
    new_baselines = []
    for baseline in baselines:
        x, y, w = baseline
        new_baseline = (x, y+delta, w)
        new_baselines.append(new_baseline)
    
    return new_baselines


def _extend_img(img, total_width):
    height, actual_width = img.shape
    diff = total_width - actual_width
    
    if diff > 0:
        padding = np.zeros((height, diff))
        img = np.append(img, padding, axis=1)

    return img


def _join_lines(imgs, annotations, baselines, config):
    result = None
    result_annotations = []
    result_baselines = []
    spacing_mean = config["Page"]["linespacemean"]
    spacing_sigma = config["Page"]["linespacesigma"]
    result_height = 0
    
    if len(imgs) > 0:
        max_width = max([i.shape[1] for i in imgs])

        for image, annotation, baseline in zip(imgs, annotations, baselines):
            img = _extend_img(image, max_width)
            
            if result is None:
                result = img
                result_annotations = annotation
                if type(baseline) is list:
                    result_baselines = baseline
                else:
                    result_baselines = [baseline]
            else:
                spacing = int(np.random.normal(spacing_mean, spacing_sigma))
                padding = np.zeros((spacing, max_width))

                result_height = result.shape[0]
                for increased_annotation in _annotations_increase_vertical_position(annotation, result_height + spacing):
                    result_annotations.append(increased_annotation)

                result = np.concatenate((result, padding, img))
                
                if type(baseline) is list:
                    for increased_baseline in _baseline_increase_vertical_position(baseline, result_height + spacing):
                        result_baselines.append(increased_baseline)
                else:
                    x, y, w = baseline
                    result_baselines.append((x, y + result_height + spacing, w))

    return result, result_annotations, result_baselines


def render_page(font, text, config):
    target_number_of_lines = config["Page"]["numberoflines"]
    paragraph_counter = 0
    number_of_lines = 0
    last_paragraph_chars = 0

    par_imgs = []
    par_annotations = []
    par_baselines = []

    while number_of_lines < target_number_of_lines and len(text) > 0:
        image, annotations, baselines = _render_paragraph(font, text[0], config, target_number_of_lines-number_of_lines)
      
        par_imgs.append(image)
        par_annotations.append(annotations)
        par_baselines.append(baselines)

        number_of_lines += len(baselines)
        
        last_paragraph_chars = len(annotations)

        if number_of_lines < target_number_of_lines:
            text = text[1:]

    if len(text) > 0 and last_paragraph_chars < len(text[0]):
        text[0] = text[0][last_paragraph_chars + 1:]
    else:
        text = text[1:]
    
    #print(par_baselines)
    page, annotations, baselines = _join_lines(par_imgs, par_annotations, par_baselines, config)
    
    return _grayscale_to_rgba(page), annotations, baselines, text

def _render_paragraph(font, text, config, max_lines=float('inf')):
    # calculation of bounding box is not correct,
    # therefore this is constant which is always 
    # added to calculated width and height and 
    # half of it is substracted from baseline 
    # (baseline is measured from bottom edge of 
    # the image)
    SPACE_PADDING=50

    font_size = 0
    try:
        font_size = config["FontSizes"][font]
    except KeyError:
        font_size = _calculate_font_size(font, config)
        config["FontSizes"][font] = font_size
    
    config["CurrentTransformations"] = _generate_transformations(text, config)

    face = Face(font)
    face.set_char_size(font_size)

    text_copy = text

    lines = []
    lines_imgs = []
    lines_annotations = []
    baselines = []
    
    while len(text_copy) > 0 and len(lines) < max_lines:
        line = _get_next_line(face, text_copy, config)
        width, height, baseline = _calculate_bounding_box(face, line, config);

        line_img = np.zeros((height + SPACE_PADDING, width + SPACE_PADDING), dtype=np.ubyte)
        line_char_positions = _render_text_to_bitmap(face, line, width, height, baseline - SPACE_PADDING/2, line_img, config)

        line_img, removed = _remove_trailing_space(line_img)
        line_char_positions = _update_positions(line_char_positions, line_img.shape, removed)
        
        line_annotations = zip(line, line_char_positions)
        
        lines.append(line)
        lines_imgs.append(line_img)
        lines_annotations.append(line_annotations)
        baselines.append(height - (baseline + SPACE_PADDING/2 - removed[1]) - 1)

        # test_img = _grayscale_to_rgba(line_img)
        # file_helper.write_image(test_img, config["Common"]["outputs"] + "/" + line + ".png")
        # annotated_img = image_helper.draw_annotations(test_img, line_annotations, [baselines[-1]], [255,0,0,0], [0,255,0,0])
        # file_helper.write_image(annotated_img, config["Common"]["outputs"] + "/" + line + "_annotated.png")

        number_of_chars = len(line)

        text_copy = text_copy[number_of_chars + 1:]
        config["CurrentTransformations"]["rotations"] = config["CurrentTransformations"]["rotations"][number_of_chars + 1:]
        config["CurrentTransformations"]["translations"] = config["CurrentTransformations"]["translations"][number_of_chars + 1:]
        config["CurrentTransformations"]["spaces"] = None

    correct_baselines = []
    for line_img, baseline in zip(lines_imgs, baselines):
        correct_baselines.append((0, baseline, line_img.shape[1]))

    #print(lines_annotations)
    img, annotations, baselines = _join_lines(lines_imgs, lines_annotations, correct_baselines, config)
    
    # test_img = _grayscale_to_rgba(img)
    # file_helper.write_image(test_img, config["Common"]["outputs"] + "/result.png")
    # annotated_img = image_helper.draw_annotations(test_img, annotations, baselines, [255,0,0,0], [0,255,0,0])
    # file_helper.write_image(annotated_img, config["Common"]["outputs"] + "/result_annotated.png")
    
    return img, annotations, baselines


def _calculate_font_size(font, config):
    font_size = 2000
    text = "abcdefghijklmnopqrstuvwxyz"
    height_epsilon = 2

    face = Face(font)
    face.set_char_size(font_size)

    _, height, _= _calculate_bounding_box(face, text, config);

    target_height = config["Page"]["lineheight"]
    lower_bound = target_height - height_epsilon
    upper_bound = target_height + height_epsilon

    while not (lower_bound <= height <= upper_bound):
        if height < target_height:
            font_size += 1
        else:
            font_size -= 1
        
        face.set_char_size(font_size)
        _, height, _= _calculate_bounding_box(face, text, config)

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
    top = 0
    bottom = 0
    left = 0
    right = 0

    while sum(img[:, 0]) == 0:
        img = np.delete(img, 0, 1)
        left += 1

    while sum(img[:, -1]) == 0:
        img = np.delete(img, -1, 1)
        right += 1

    while sum(img[-1, :]) == 0:
        img = np.delete(img, -1, 0)
        bottom += 1
        
    while sum(img[0, :]) == 0:
        img = np.delete(img, 0, 0)
        top += 1

    return img, (top, bottom, left, right)
