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
from sympy import Point
import sys


log = getLogger()


@misc.exit_on_exception(1)
@misc.debug_on_exception([Exception])
def generate(args, config, storage, **kwargs):
    log.debug('Function generate reached')
    log.debug(f'layout: {args.layout}')

    # load necessary data and preprocess
    layout_name = ' '.join(args.layout)
    layout = storage.layouts.get(layout_name)
    background_name, background = storage.backgrounds.random_pair()
    # font_name = list(storage.fonts.items())[0][0]
    font_name, _ = storage.fonts.random_pair()

    height, width = config['Page']['height'], config['Page']['width']
    background = presh.ensure_image_shape(background, (height, width))

    layout.fit_to_region((Point(0, 0), Point(width-1, height-1)))
    log.debug('layout fitted onto region')
    # breakpoint()

    renderer = text_renderer.Renderer(storage.fonts, 15*64)
    comp = compositor.Compositor(config, renderer)

    # render image
    img = comp.compose_image(background, font_name, storage.text, iter(layout))
    log.info(f'Image composed with font: {font_name}, '
             f'background: {background_name} ({width} x {height})')

    # view result image
    im_viewer = presh.ImageViewer.from_config(config)
    img = presh.np_array_to_img(img)
    im_viewer.view_img(img)

    sys.exit(0)
