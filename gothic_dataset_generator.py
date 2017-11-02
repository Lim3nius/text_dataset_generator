
import sys
from PIL import Image
from freetype import *
import numpy as np

def calculate_bounding_box(face, text):
    slot = face.glyph
    width, height, baseline, width_add = 0, 0, 0, 0
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

    return width, height, baseline


def render_text_to_bitmap(face, text, width, height, baseline, image_array):
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
        image_array[y:y+h,x:x+w] += np.array(bitmap.buffer, dtype='ubyte').reshape(h,w)
        characters_position.append(x + w / 2)
        x += (slot.advance.x >> 6)
        previous = c

    return characters_position


def render_text(font, text, image_name, font_size=32):
    font_size_coeficient = 64
    face = Face(font)
    face.set_char_size(font_size * font_size_coeficient)

    width, height, baseline = calculate_bounding_box(face, text);

    img = np.zeros((height,width), dtype=np.ubyte)

    positions = render_text_to_bitmap(face, text, width, height, baseline, img)

    annotations = zip(text, positions)
    
    im = Image.fromarray(255 - img)
    im.save(image_name)

    return img, annotations


def parse_arguments():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', help='Input text file name.')
    parser.add_argument('-o', '--output', help='Output directory name.')
    parser.add_argument('-f', '--font', help='Font file name.')
    parser.add_argument('-s', '--size', help='Font size.')
    args = parser.parse_args()
    return args


def read_file(file_name):
    content = []

    with open(file_name, "r") as f_read:
        for line in f_read:
            content.append(line.rstrip())
    
    return content


def write_annotation_file(annotations, file_name):
    with open(file_name, "w") as f_write:
        for annotation in annotations:
            f_write.write(str(annotation) + "\r\n")


def create_directory_if_not_exists(dir_name):
    import os
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

def main():
    args = parse_arguments()
    content = read_file(args.input)
    create_directory_if_not_exists(args.output)
        
    for index, line in enumerate(content):
        img, annotations = render_text(args.font, line, args.output + "/image_" + str(index) + ".png", int(args.size))
        write_annotation_file(annotations, args.output + "/image_" + str(index) + ".txt")

    return 0;

if __name__ == "__main__":
    sys.exit(main())
