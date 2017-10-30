
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
    characters_position = []
    slot = face.glyph
    x, y = 0, 0
    previous = 0
    for c in text:
        face.load_char(c)
        bitmap = slot.bitmap
        top = slot.bitmap_top
        left = slot.bitmap_left
        w,h = bitmap.width, bitmap.rows
        y = height-baseline-top
        kerning = face.get_kerning(previous, c)
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


def main():
    img, annotations = render_text("Vera.ttf", "Ahoj, světe!", "image.png")

    for char_and_pos in annotations:
        print(char_and_pos)    

    return 0;

if __name__ == "__main__":
    sys.exit(main())
