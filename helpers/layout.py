#!/usr/bin/env python3

'''
File: layout.py
Author: Tomáš Lessy
Email: lessy.mot@gmail.com
Github: https://github.com/lim3nius
Description: This module contains helper classes and functions for
    layout parsing and manipulation
'''


import yaml
import math
import random
from logging import getLogger
from sympy import Point
from typing import Tuple, Union, List, Dict
# import enum
import itertools

from helpers import misc
log = getLogger()


class RegionError(Exception):
    '''Exception representing that problem occured inside Region class'''
    pass


class RegionConfiguration:
    '''RegionConfiguration represents yaml file containing multiple layouts'''
    def __init__(self, *, regions=dict()):
        self.regions = regions

    def __repr__(self) -> str:
        s = '{' + ', '.join([f'"{k}"' for k in self.regions.keys()]) + '}'
        return 'RegionConfiguration: ' + s

    def get(self, key):
        return self.regions[key]

    @staticmethod
    @misc.debug_on_exception([Exception])
    def from_dict(di: Dict):
        regions = {}

        for r in di.get('layouts'):
            r = Region.from_dict(r)

            if not r.name:
                raise ValueError('Top level layout has to have name')

            if regions.get(r.name):
                raise ValueError(
                    f'Found layout name "{r.name}" for second time!')

            regions[r.name] = r

        return RegionConfiguration(regions=regions)


class Region:
    '''
    Region class represent single region region containing lines of text
    or divided into multiple other regions

    allows of iteration through regions returning bounding boxes for specified
    line numbers
    '''
    def __init__(self, width=100.0, height=100.0,
                 *, padding=0, line_height=0, content=List[str],
                 columns: List[int], rows: List[int], name: str = ''):

        self.name = name
        self.line_height = line_height
        self.width = width
        self.height = height

        if name == 'Blank':
            return

        self.padding = padding
        self.columns = columns
        self.rows = rows
        self.content = content
        self.box = None

        if columns is not None and rows is not None:
            raise RegionError('Columns and Rows defined, but not supported')

    def deep_copy(self):
        if self.name == 'Blank':
            return Region(self.width, self.height, name=self.name,
                          columns=None, rows=None)

        if self.content:
            content = [get_content(c) for c in self.content]
        else:
            content = None
        return Region(self.width, self.height,
                      padding=self.padding, line_height=self.line_height,
                      content=content, columns=self.columns,
                      rows=self.rows, name=self.name)

    def __repr__(self) -> str:
        if self.name:
            return self.name
        else:
            return f'<Region[width:{self.width}, height:{self.height}>'

    @staticmethod
    @misc.debug_on_exception([Exception])
    def from_dict(d):
        lh = d.get('line_height', 0)
        name = d.get('name', '')

        cols = d.get('columns', '')
        rows = d.get('rows', '')

        # TODO: make it nicer
        if '-' in cols:
            l0, l1 = map(int, cols.split('-'))
            cols = list(range(l0, l1))
        elif cols != '':
            cols = [int(cols)]
        else:
            cols = None

        if '-' in rows:
            l0, l1 = map(int, rows.split('-'))
            rows = list(range(l0, l1))
        elif rows != '':
            rows = [int(rows)]
        else:
            rows = None

        con = d.get('content', None)
        padding = d.get('padding', 0)

        try:
            reg = Region(d['width'], d['height'],
                         line_height=lh, content=con, columns=cols, rows=rows,
                         name=name, padding=padding)
        except RegionError as e:
            log.error(f'Unable to parse region "{name}" because: {e}')
            raise e

        return reg

    def compute_real_val(self, v, dimension):
        if isinstance(v, int):
            return v
        elif isinstance(v, float) and (0 < v <= 1.0):
            return math.floor(dimension * v)
        else:
            raise RegionError('invalid dimension value given')

    def fit_to_region(self, region: Tuple[Point, Point]):
        if self.name == 'Blank':
            return

        top_left, bot_right = region
        if self.padding:
            top_left += Point(self.padding, self.padding)
            bot_right -= Point(self.padding, self.padding)

        self.box = (top_left, bot_right)

        if self.line_height > 0:
            return

        def val(low, up, val):
            size = up - low + 1
            if isinstance(val, float):
                val = math.ceil(size * val)

            val -= 1
            return (low, low + val) if low + val < size else (low, up)

        if self.columns:
            pick_cols = random.choice(self.columns)
            # content_idx = random.choice(self.content)
            # content = get_content(self.content[content_idx])  # Get actual Region object
            width_intervals = divide_interval(
                top_left.x, bot_right.x, [1/pick_cols for _ in range(pick_cols)])

            height_intervals = [val(top_left.y, bot_right.y, self.height)
                                for r in range(pick_cols)]
            selected_content = [get_content(str(random.choice(self.content))) for _ in range(pick_cols)]
        elif self.rows:
            pick_rows = random.choice(self.rows)
            # content_idx = random.choice(self.content)
            # content = get_content(self.content[content_idx])  # Get actual Region object
            height_intervals = divide_interval(
                top_left.y, bot_right.y, [1/pick_rows for _ in range(pick_rows)])

            width_intervals = [val(top_left.x, bot_right.x, self.width)
                               for r in range(pick_rows)]
            selected_content = [get_content(str(random.choice(self.content))) for _ in range(pick_rows)]

        # selected content means selected layouts for subregions
        self.selected_content = selected_content

        zp = zip(selected_content, width_intervals, height_intervals)
        for (reg, wi, hi) in zp:
            reg.fit_to_region((Point(wi[0], hi[0]), Point(wi[1], hi[1])))

    def get_text_regions_generators(self) -> List[List[Point]]:
        if self.line_height != 0:
            return [[(Point(self.box[0].x, y),
                     Point(self.box[1].x, y + self.line_height - 1))
                    for y in range(
                        self.box[0].y, self.box[1].y, self.line_height)
                    if y + self.line_height - 1 <= self.box[1].y]]
        else:
            return [i for i in self.selected_content]

    def __iter__(self):
        '''
        Returns iterator which returns Tuple(Point, Point)
        representing TextLine with bounding box created from
        top left and right bottom points
        '''
        if self.name == 'Blank':
            return iter([])

        if not self.box:
            raise RegionError(
                'Have to call "fit_to_region", before iterating')

        if self.line_height:
            return ((Point(self.box[0].x, y),
                     Point(self.box[1].x, y + self.line_height - 1))
                    for y in range(
                        self.box[0].y, self.box[1].y, self.line_height)
                    if y + self.line_height - 1 <= self.box[1].y)
        else:
            line_iters = [i for i in self.selected_content]
            return itertools.chain(*line_iters)


Interval = Tuple[int, int]


def divide_interval(lower: int, upper: int,
                    chunks: List[Union[int, float]]) -> List[Interval]:
    '''
    divide_interval divides interval given by lower and upper bound
    (inclusively) into subintervals of specified length.
    If interval isn't big enough, other subinterval are added of length 0
    right at the end of given interval

    :returns: List of tuples containing lower boundary (inclusively) and upper
    boundary (exclusively)
    '''
    if lower < 0 or upper < 0 or lower > upper:
        raise ValueError('Invalid interval range')

    length = upper - lower + 1
    s = lower
    res = []

    for chunk in chunks:
        # if interval is already divided
        if s >= upper:
            res.append((upper, upper))
            continue

        # handle percentage of length
        if isinstance(chunk, float):
            chunk = math.ceil(length * chunk)

        if s + chunk - 1 <= upper:
            t = s + chunk - 1
            res.append((s, t))
            s = t + 1

        elif s + chunk > upper:
            res.append((s, upper))
            s = upper

    return res


layouts: Dict[str, RegionConfiguration] = {}


def init_layouts(la: Dict[str, RegionConfiguration]):
    global layouts
    layouts = la


def get_content(content_name: str) -> RegionConfiguration:
    global layouts
    log.info(f'Layouts content: {layouts}')
    layout = layouts[content_name]
    return layout.deep_copy()


def load_layouts(path: str) -> RegionConfiguration:
    '''load_layout loads region layouts specified in file pointed by path'''
    with open(path, 'r') as f:
        d = yaml.safe_load(f)

    regions = RegionConfiguration.from_dict(d)
    global layouts
    layouts = regions.regions
    return regions
