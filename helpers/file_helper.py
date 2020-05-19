from PIL import Image
import numpy as np
import string
import os

from typing import Iterable, List
from pathlib import Path
from random import choice
from freetype import Face


def read_file(file_name, words=False):
    content = []

    with open(file_name, "r") as f_read:
        for line in f_read:
            if words:
                splitted_words = line.rstrip().split()
                for single_word in splitted_words:
                    content.append(single_word.translate(
                        None, string.punctuation))
            else:
                content.append(line.rstrip())

    return content


def write_annotation_file(annotations, baselines, file_name):
    with open(file_name, "w") as f_write:
        for annotation in annotations:
            f_write.write(str(annotation) + "\r\n")

        for baseline in baselines:
            f_write.write(str(baseline) + "\r\n")


def write_file(content, file_name):
    with open(file_name, "w") as f_write:
        if type(content) is list:
            for line in content:
                f_write.write(str(line) + "\r\n")
        else:
            f_write.write(str(content))


def create_directory_if_not_exists(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)


def write_image(img, path):
    im = Image.fromarray(img)
    if im.mode != 'RGBA':
        im = im.convert('RGBA')

    im.save(path)


def read_image(path):
    img = Image.open(path)
    img = np.array(img)

    if img.shape[2] == 4:
        img = img[:, :, :3]

    return img


def load_all_images(dir_name):
    images = []
    image_files = [f for f in os.listdir(dir_name)
                   if os.path.isfile(os.path.join(dir_name, f)) and
                   (f.endswith(".jpg") or f.endswith(".png"))]

    for file in image_files:
        images.append(read_image(os.path.join(dir_name, file)))

    return images


def load_all_fonts(dir_name):
    fonts = [os.path.join(dir_name, f) for f in os.listdir(dir_name)
             if os.path.isfile(os.path.join(dir_name, f)) and
             f.lower().endswith(".otf") or f.lower().endswith(".ttf")]
    return fonts


class LazyLoader(dict):
    """
    LazyLoader is dict which loads key data lazily.
    When item is accessed first time, loader_fn is called with argument being
    the key
    """
    def __init__(self, collection: Iterable, loader_fn):
        self.collection = set(collection)
        self.loader_fn = loader_fn

    def __getitem__(self, key):
        try:
            v = dict.__getitem__(self, key)
        except KeyError:
            v = self.loader_fn(key)
            dict.__setitem__(self, key, v)

            if key not in self.collection:
                self.collection.add(key)

        return v

    def keys(self) -> Iterable:
        return list(self.collection)

    def load(self):
        for k in self.collection:
            try:
                v = self.loader_fn(k)
            except Exception as e:
                raise Exception(f'Error occured during loading {k} -> {e}')
            dict.__setitem__(self, k, v)

    def __repr__(self):
        return self.collection.__repr__()

    def __len__(self):
        return len(self.collection)

    def random_pair(self):
        c = choice(list(self.collection))
        return (c, self[c])


def load_images(dir_name) -> LazyLoader:
    p = Path(dir_name)
    if not p.exists() or not p.is_dir():
        raise Exception('Invalid images directory "{dir_name}"')

    images = []
    for ext in ['jpg', 'png']:
        images.extend(list(p.glob('*.'+ext)))

    return LazyLoader(images, lambda i: read_image(i))


def load_fonts(dir_name: str) -> LazyLoader:
    p = Path(dir_name)
    if not p.exists() or not p.is_dir():
        raise Exception('Invalid font directory "{dir_name}"')

    fonts = []
    for ext in ['otf', 'ttf']:
        fonts.extend(list(map(lambda e: e.as_posix(), p.glob('*.'+ext))))

    return LazyLoader(fonts, lambda f: Face(f))


def load_font(path: str) -> Face:
    return Face(path)
