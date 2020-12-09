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
    layout_name = ' '.join(args.layout)
    layout = storage.layouts.get(layout_name)
    background_name, background = storage.backgrounds.random_pair()
    font_name, _ = storage.fonts.random_pair()

    height, width = config['Page']['height'], config['Page']['width']
    background = presh.ensure_image_shape(background, (height, width))
    page = pageXML.PageLayout(id='nafjasafn', page_size=(height, width))

    layout.fit_to_region((Point(0, 0), Point(width-1, height-1)))
    log.debug('layout fitted onto region')

    renderer = text_renderer.Renderer(storage.fonts, 15*64)
    comp = compositor.Compositor(config, renderer)

    # regions = list(iter(layout))
    regions = layout.get_text_regions_generators()
    log.info(f'Computed regions: {regions}')
    # breakpoint()

    # render image
    img, line_imgs = comp.compose_image(background, font_name, storage.text, regions)

    log.info(f'Image composed with font: {font_name}, '
             f'background: {background_name} ({width} x {height})')

    page.regions = []
    for (i1, region) in enumerate(line_imgs):
        text_lines = []
        for (i2, line) in enumerate(region):
            t = pageXML.TextLine(
                id=str(i2), baseline=[[line.baseline, int(line.point.x)]],
                polygon=gen_bounding_polygon(line.bounding_box),
                transcription=line.text)
            text_lines.append(t)

        rl = pageXML.RegionLayout(id=str(i1), polygon=[[42.0, 42.0], [3,14, 2,71]])
        rl.lines = text_lines
        # rl = pageXML.RegionLayout(id=str(i), polygon=gen_bounding_polygon(r))
        page.regions.append(rl)

    # view result image
    im_viewer = presh.ImageViewer.from_config(config)
    img = presh.np_array_to_img(img)
    im_viewer.view_img(img)
    breakpoint()
    # import IPython; IPython.embed()
    print(page.to_pagexml_string())

    sys.exit(0)
