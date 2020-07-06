"""
File: baseline_helper.py
Author: Tomas Lessy
Email: lessy.mot@gmail.com
Github: https://github.com/lim3nius
Description: Main baseline rendering module
"""

import sys
from logging import getLogger
from typing import List, Tuple
from copy import deepcopy
import numpy as np
from PIL import Image
from math import floor

from sympy import Line, Point, solve, symbols

from pero_ocr.document_ocr.layout import PageLayout
from helpers.text_renderer import Renderer
# import helpers.image_helper as imgh
import helpers.presentation_helper as presh
# from helpers.logger import log_function_call
from helpers.misc import exit_on_exception

log = getLogger()

# Point = Tuple[int, int]


class Compositor:
    """docstring for Compositor"""
    def __init__(self, config):
        self.config = config

    def point_on_baseline(x_pos, baseline_segment):
        """
        For new character with known X coordinate
        computes Y position
        """
        l = Line(Point(baseline_segment[:2]), Point(baseline_segment[2:]))
        x = symbols('x')
        s = solve(l.equation(x=x).subs({'x': x_pos}))
        return s[0]

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


def calculate_polygon_outer_bbox(
        points: List[List[int]]) -> List[Point]:
    min_x = min(points, key=lambda l: l[0])[0]
    max_x = max(points, key=lambda l: l[0])[0]
    min_y = min(points, key=lambda l: l[1])[1]
    max_y = max(points, key=lambda l: l[1])[1]
    return [(min_x, min_y), (max_x, max_y)]


def calculate_inner_bbox_height(
        points: List[List[int]]) -> int:
    y_coords = [l[1] for l in points]
    y_coords = sorted(y_coords)

    biggest_diff = 0
    for i in range(len(y_coords)):
        if i == len(y_coords) - 1:
            break

        d = abs(y_coords[i] - y_coords[i+1])
        if d > biggest_diff:
            biggest_diff = d

    return biggest_diff


def calculate_bbox_height(min_p: Point, max_p: Point) -> int:
    return max_p[1] - min_p[1]


def load_page(path: str) -> PageLayout:
    try:
        page = PageLayout(file=path)
    except Exception as e:
        log.error(f'Exception occured during parsing file: {path}'
                  f', {e}')
        return None

    log.warn(f'page size {page.page_size}')
    if page.page_size == (0, 0):
        log.error(f'Invalid file "{path}" has been given')
        return None

    return page


def show_baselines(page: PageLayout, img: Image.Image) -> Image.Image:
    for r in page.regions:
        reg_color = presh.random_color()
        for l in r.lines:
            img = presh.write_points(img, '#FF0000', l.baseline)
            img = presh.write_polygon(img, '#00FF00', l.polygon)
            bbox = calculate_polygon_outer_bbox(l.polygon)
            img = presh.write_rectangle(img, reg_color, bbox)
    return img


@presh.view_result_decorator('/tmp/tdg/', 'feh')
def rerender_page(page: PageLayout, renderer: Renderer,
                  font: str, background: np.array) -> np.array:
    pheight, pwidth = page.page_size
    background = presh.ensure_image_shape(background, (pheight, pwidth))

    log.info('Page rendering started')
    c = Compositor(None)

    for r in page.regions:

        if r.lines is None or len(r.lines) == 0:
            continue
        # assuming all TextLines in region are of same font size
        longest_line = max(r.lines, key=lambda e: len(e.transcription))
        outer_bbox = calculate_polygon_outer_bbox(longest_line.polygon)
        region_line_height = floor(calculate_bbox_height(*outer_bbox) * 0.8)
        log.info(f'For region {r.id} selected '
                 f'longest_line: "{longest_line.transcription}"'
                 f', with height: {region_line_height}')
        region_font_size = renderer.calculate_font_size(
            font, region_line_height,
            target_width=outer_bbox[1][0] - outer_bbox[0][0],
            text=longest_line.transcription)

        for l in r.lines:
            text = l.transcription

            poly = calculate_polygon_outer_bbox(l.polygon)
            line_width = poly[1][0] - poly[0][0]
            line_height = floor(calculate_bbox_height(*poly) * 0.8)
            log.debug(f'Line height: {line_height}')

            font_size = region_font_size
            # In case title is in TextRegion with rest of text
            if line_height >= 1.75 * region_line_height:
                font_size = renderer.calculate_font_size(
                    font, line_height, target_width=line_width, text=text)

            log.debug(f'Rendering text: "{text}"')
            # make sure that all cahracters are rendered and fits into
            # bounding box
            recalc = 0
            while True:
                text_img = renderer.render_line(
                    text, font, font_size, line_width)
                # text_img = renderer.draw(text, font, region_font_size)
                height, width = text_img.bitmap.shape[:2]

                if width <= line_width:
                    break
                else:
                    font_size = renderer.calculate_font_size(
                        font, line_height - recalc, target_width=line_width,
                        text=text)
                    log.info('recalculating font size')
                    log.info(f'received h, w: {height}, {width},'
                             f'wanted: {line_height}, {line_width}')
                    log.info(f'line height: {line_height - recalc}')
                    recalc += 1

            # let's say first point is baseline
            top_left = deepcopy(l.baseline[0])
            top_left[1] -= text_img.baseline

            if width > line_width:
                log.warn(f'Cropping: {width} > {line_width}')
                text_img.bitmap = text_img.bitmap[:, :line_width, :]
                width = line_width

            log.debug(f'top left: {top_left}')
            log.debug(f'text image shape {text_img.bitmap.shape}')
            c.place_text_on_background(text_img.bitmap, background, top_left)

    log.info('Page rendering finished')
    return background


@exit_on_exception(1)
def main(args, config, storage):
    """
    main handler for generating images from baselines
    """

    path = args.path
    page = load_page(path)
    renderer = Renderer(storage.fonts, 14 * 64)
    viewer = presh.init_global_viewer(config['Common']['viewer'],
                                      config['Common']['tempdir'],
                                      config['Common']['imageformat'])

    background_name, background = storage.backgrounds.random_pair()
    log.info(f'Background "{background_name}" loaded')

    pheight, pwidth = page.page_size
    log.debug(f'Background dimensions {background.shape}')
    background = presh.ensure_image_shape(background, (pheight, pwidth))
    log.debug(f'Background dimensions after shape change {background.shape}')

    font = list(storage.fonts.items())[0][0]
    log.debug(f'Using font: {font}')
    gen_page = rerender_page(page, renderer, font, np.copy(background))
    gen_page = show_baselines(page, presh.np_array_to_img(gen_page))
    viewer.view_img(gen_page)

    img = presh.np_array_to_img(background)
    log.debug(f'Background dimensions {img.size}')

    img = show_baselines(page, img)

    log.debug(f'Result image dimensions -> {img.size}')
    background = path.split('.')[0] + '.jpg'
    background = presh.load_image(background)

    img = presh.compare_images(img, background, 0.5)

    viewer.view_img(img)

    log.info('Baseline helper execution ended')
    sys.exit(0)
