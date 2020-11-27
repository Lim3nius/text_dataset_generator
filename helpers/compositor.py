#!/usr/bin/env python3

"""
File: compositor.py
Author: Tomáš Lessy
Email: lessy.mot@gmail.com
Github: https://github.com/lim3nius
Description: his module containg Compositor class used to compose
    all those little parts into one image
"""

import numpy as np
from sympy import Point
from typing import Tuple

from helpers import misc
from helpers import text_renderer as trenderer


class Compositor:
    """docstring for Compositor"""
    def __init__(self, config, renderer):
        self.config = config
        self.renderer = renderer

    @misc.debug_on_exception([Exception])
    def place_text_on_background(self, text_img: np.ndarray,
                                 background: np.ndarray,
                                 point: Tuple[int, int]):
        '''
        place_text_on_background places given text strip on background.
        Position is determined by given point, representing top left corner
        of text on background

        :text_img: np.ndarray representing text strip. Shape should be
            (height, width, 3) and text pixels should have 0 values
        :background: np.ndarray of shape (height, width, 3), containing
            background data.
        :point: coordinates of left top corner of text strip on background,
            data order is (width, height)
        '''

        h, w = text_img.shape[:2]
        pw, ph = point
        # invertion because text black color has value 0
        alpha_text = 1.0 - (text_img[:, :, -1] / 255.0)
        alpha_back = 1.0 - alpha_text
        for chan in range(3):
            background[ph:ph+h, pw:pw+w, chan] = (
                alpha_text * text_img[:, :, chan] +
                alpha_back * background[ph:ph+h, pw:pw+w, chan])

    @misc.debug_on_exception([Exception])
    def compose_image(self, background, font, text_prov,
                      lines_prov) -> trenderer.AnnotatedTextImage:
        img = np.copy(background)
        for line in lines_prov:
            ln_height = line[1].y - line[0].y + 1
            ln_width = line[1].x - line[0].x + 1
            font_size = self.renderer.calculate_font_size(font, ln_height)
            text = self.select_text_for_line(ln_width, text_prov, font,
                                             font_size)
            # text_img, _ = self.renderer.render_line(text, font, font_size,
            #                                         ln_width)
            text_img = self.renderer.draw(text, font, font_size)
            self.place_text_on_background(text_img.bitmap, img,
                                          (line[0].x, line[0].y))

        return img

    @misc.debug_on_exception([Exception])
    def select_text_for_line(self, line_width: int, word_provider,
                             font: str, font_size: int) -> str:
        '''select_text_for_line returns text which fits into given line'''
        text = ''
        while True:
            word = word_provider.get_word()
            prev_text = text

            if text == '':
                text = word
            else:
                text += ' ' + word

            face = self.renderer.get_face(font)
            face.set_char_size(font_size)
            width, _, _ = self.renderer.calculate_bbox(face, text)
            if width >= line_width:
                word_provider.accept_word()
                return prev_text

            word_provider.accept_word()
