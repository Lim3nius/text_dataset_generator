
import sys
from PIL import Image
from freetype import *
import numpy as np

def calculate_bounding_box(face, text):
    characters_position = []
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
        width_add = (slot.advance.x >> 6) + (kerning.x >> 6)
        characters_position.append(width + width_add / 2)
        width += width_add
        previous = c

    return width, height, baseline, characters_position


def render_text(face, text, width, height, baseline, image_array):
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
        x += (slot.advance.x >> 6)
        previous = c


def main():
    face = Face('./Vera.ttf')
    text = 'Hello World !'
    face.set_char_size( 48*64 )

    width, height, baseline, positions = calculate_bounding_box(face, text);

    img = np.zeros((height,width), dtype=np.ubyte)

    render_text(face, text, width, height, baseline, img)

    annotations = zip(text, positions)

    for t in annotations:
        print(t)

    im = Image.fromarray(img)
    im.save('test.png')

    return 0;

if __name__ == "__main__":
    sys.exit(main())
