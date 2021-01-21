#!/usr/bin/env python3

"""
File: generator.py
Author: Tomáš Lessy
Email: lessy.mot@gmail.com
Github: https://github.com/lim3nius
Description: Module containing main routine for new generation of images
"""

from helpers import misc
from logging import getLogger
import helpers.presentation_helper as presh
from helpers import compositor
from helpers import text_renderer
import sys
import pero_ocr.document_ocr.layout as pageXML
from sympy import Point
from typing import List, Tuple
from itertools import chain

import numpy as np


log = getLogger()


# Helper function to generate bounding polygon from bounding box
def gen_bounding_polygon(bbox: Tuple[Point, Point]) -> List[List[int]]:
    return [list(map(float, p)) for p in [bbox[0], Point(bbox[1].x, bbox[0].y),
                                          bbox[1], Point(bbox[0].x, bbox[1].y)]]


@misc.exit_on_exception(1)
@misc.debug_on_exception([Exception])
def generate(args, config, storage, **kwargs):
    log.debug('Function generate reached')
    log.debug(f'layout: {args.layout}')

    # load necessary data and preprocess
    renderer = text_renderer.Renderer(storage.fonts, 15*64)
    print(text_renderer.default_renderer)
    layout_name = ' '.join(args.layout)
    layout = storage.layouts.get(layout_name)
    background_name, background = storage.backgrounds.random_pair()
    font_name, _ = storage.fonts.random_pair()

    height, width = config['Page']['height'], config['Page']['width']
    background = presh.ensure_image_shape(background, (height, width))
    page = pageXML.PageLayout(id='nafjasafn', page_size=(height, width))

    layout.fit_to_region(Point(0, 0), Point(width-1, height-1))
    log.debug('layout fitted onto region')

    # renderer = text_renderer.Renderer(storage.fonts, 15*64)
    # print(text_renderer.default_renderer)
    comp = compositor.Compositor(config, renderer)

    regions = list(layout)
    log.info(f'Computed regions: {regions}')

    # render image
    img = background
    for data_reg in regions:
        img, line_imgs = comp.compose_image(img, data_reg.font, storage.text,
                                            data_reg, font_size=data_reg.font_size)

    log.info(f'Image composed with font: {font_name}, '
             f'background: {background_name} ({width} x {height})')

    # generate page regions description
    # page.regions = []
    # for (i1, region) in enumerate(line_imgs):
    #     text_lines = []
    #     minx, miny = 10**9, 10**9
    #     maxx, maxy = 0, 0
    #     for (i2, line) in enumerate(region):
    #         t = pageXML.TextLine(
    #             id=str(i2), baseline=line.baseline_with_context(),
    #             polygon=gen_bounding_polygon(line.bounding_box),
    #             transcription=line.text)
    #         text_lines.append(t)
    #         minx = min(int(line.point.x), minx)
    #         maxx = max(int(line.point.x), maxx)
    #         miny = min(int(line.point.y), miny)
    #         maxy = max(int(line.point.y), maxy)

    #     rl = pageXML.RegionLayout(id=str(i1), polygon=[[minx, miny], [minx, maxy], [maxx, maxy], [minx, maxy]])
    #     rl.lines = text_lines
    #     # rl = pageXML.RegionLayout(id=str(i), polygon=gen_bounding_polygon(r))
    #     page.regions.append(rl)

    # view result image
    im_viewer = presh.ImageViewer.from_config(config)
    img = presh.np_array_to_img(img)
    im_viewer.view_img(img)
    print(page.to_pagexml_string())
    with open('/tmp/tdg/page.xml', 'w') as fd:
        fd.write(page.to_pagexml_string())

    sys.exit(0)
