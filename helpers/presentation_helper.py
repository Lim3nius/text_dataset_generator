"""
File: presentation_helper.py
Author: Tomas Lessy
Email: lessy.mot@gmail.com
Github: https://github.com/lim3nius
Description: Helper for showing intermediate results
"""

from logging import getLogger
from typing import List, Tuple
from subprocess import run
from random import choice
from string import ascii_letters
from functools import wraps

from PIL import Image
from PIL import ImageColor as Color
from PIL.ImageDraw import Draw
import numpy as np

from helpers.logger import log_function_call
from helpers.file_helper import write_image
from helpers.misc import debug_on_exception

Point = Tuple[int, int]
log = getLogger()


@debug_on_exception([Exception])
def compare_images(img1, img2, alpha):
    return Image.blend(img1, img2, alpha)


def show_image(img: Image.Image):
    img.show(title='Generated image', command='ristretto')


def np_array_to_img(arr: np.array) -> Image.Image:
    return Image.fromarray(arr)


def write_points(img: Image.Image, color_str: str,
                 points: List[int]) -> Image.Image:
    """Function write_points is used for showing list of points
    in image
    """

    color = Color.getrgb(color_str)
    img = img.copy()
    imc = Draw(img)
    for p in points:
        w, h = p
        imc.rectangle([(w, h), (w+3, h+3)], color, color)

    return img


def write_polygon(img: Image.Image, color_str: str,
                  points: List[List[int]]) -> Image.Image:
    pts = []
    for p in points:
        pts.extend(p)

    color = Color.getrgb(color_str)
    img = img.copy()
    imc = Draw(img)
    imc.polygon(pts, outline=color)
    return img


@log_function_call('debug')
def write_rectangle(img: Image.Image, color_str: str,
                    points: List[Point]) -> Image.Image:
    log.debug(f'Rendering rectangle')
    color = Color.getrgb(color_str)
    img = img.copy()
    imc = Draw(img)
    imc.rectangle(points, outline=color)
    return img


def load_image(path):
    img = Image.open(path)
    img = img.convert(mode='RGB')
    log.debug(f'Load image size {img.size}, mode = {img.mode}')
    return img


def ensure_image_shape(img, shape):
    """
    Simple function which clones parts of image to reach desired shape.
    In extreme case whole image, no postprocessing now
    """

    height, width = img.shape[:2]
    t_height, t_width = shape

    if t_height < height:
        log.debug('-- height')
        img = img[:t_height]
    elif t_height > height:
        log.debug('++ height')
        diff = t_height - height

        if diff > height:
            log.warn('risky height increase')
            mult = (t_height - height) // height
            diff = (t_height - height) % height
            for _ in range(mult):
                img = np.concatenate((img, img[:height]), axis=0)

        img = np.concatenate((img, img[:diff]), axis=0)

    if t_width < width:
        log.debug('-- width')
        img = img[:, :t_width]
    elif t_width > width:
        log.debug('++ width')
        diff = t_width - width
        if diff > width:
            log.warn('Risky width increase')
            mult = (t_width - width) // width
            diff = (t_width - width) % width
            for _ in range(mult):
                img = np.concatenate((img, img[:, :width]), axis=1)

        img = np.concatenate((img, img[:, :diff]), axis=1)

    return img


def view_result_decorator(save_location: str, viewer: str):
    def wrapper(f):
        def inner(*args, **kwargs):
            res = f(*args, **kwargs)

            path = ''
            if isinstance(res, Image.Image):
                fname = random_string(9)
                path = (save_location + '/' + fname + '.png')
                if res.mode != 'RGBA':
                    img = res.convert('RGBA')
                    img.save(path)
                res.save(path)

            elif (isinstance(res, np.ndarray) and len(res.shape) == 3
                  and res.shape[-1] == 3):
                fname = random_string(9)
                path = (save_location + '/' + fname + '.png')
                write_image(res, path)

            else:
                log.warn('Nothing to display')
                return res

            view_file(viewer, path)
            return res
        return inner
    return wrapper


def random_string(length: int) -> str:
    if length <= 0:
        return ''
    return ''.join([choice(ascii_letters) for _ in range(length)])


def view_file(viewer: str, file_path: str):
    res = run(args=viewer + ' ' + file_path, shell=True)

    if res.returncode != 0:
        raise Exception('view_file failed with exitcode ' + res.returncode)


class ImageViewer:
    """docstring for ClassName"""
    def __init__(self, viewer: str, tmp_dir: str):
        self.viewer = viewer
        self.tmp_dir = tmp_dir

    def view_img(self, img: Image.Image, format='png'):
        if img.mode != 'RGBA':
            img = img.convert('RGBA')

        path = self.tmp_dir + '/' + random_string(10) + '.' + format
        img.save(path)

        view_file(self.viewer, path)
