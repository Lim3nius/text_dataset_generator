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
import re
from logging import getLogger
from sympy import Point
from typing import Tuple, Union, List, Dict, Any
# import enum
import itertools

from helpers import misc
from helpers import text_renderer
log = getLogger()


class RegionError(Exception):
    '''Exception representing that problem occured inside Region class'''
    pass


class RegionConfiguration:
    '''RegionConfiguration represents yaml file containing multiple layouts'''
    def __init__(self, *,
                 regions: Dict[str, Union['DataRegion', 'LayoutRegion']] = None):
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

        for reg in di.get('layouts'):
            for cl in [LayoutRegion, DataRegion]:
                try:
                    reg = cl.from_dict(reg)
                except Exception as ex:
                    print(ex)
                    continue
                else:
                    break
            else:
                raise Exception('Unable to parse structures')

            if not reg.name:
                raise ValueError('Top level layout has to have name')

            if regions.get(reg.name):
                raise ValueError(
                    f'Found layout name "{reg.name}" for second time!')

            regions[reg.name] = reg

        return RegionConfiguration(regions=regions)


class DataRegion:
    '''
    Terminal region into which is text written, in yaml seems like normal
    rule, except on properties it have
    '''
    def __init__(self, width=1.0, height=1.0, *, name='',
                 padding=0, font_size=24*64, spacing=10, font='Free Serif', **kwargs):
        self.name = name
        self.width = width
        self.height = height
        self.padding = padding

        if isinstance(font_size, tuple):
            font_size = font_size[0]
        if not isinstance(font_size, int):
            raise Exception('WTF font size: {}'.format(font_size))
        self.font_size = font_size
        self.spacing = spacing
        self.font = font

        self.bounding_box = None  # variable for some low level shit

    def deep_copy(self) -> 'DataRegion':
        return DataRegion(**vars(self))

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> 'DataRegion':
        return DataRegion(
            data.get('width', 1.0), data.get('height', 1.0),
            name=data['name'],
            padding=data.get('padding', 0), font_size=int(data['font_size'])*64,
            font=data['font'], spacing=data['spacing'],
        )

    def fit_to_region(self, topl: Point, botr: Point):
        '''Sets bounding box onto which will this DataRegion operate'''
        self.bounding_box = (topl, botr)

    def __iter__(self):
        tmp = text_renderer.default_renderer.calculate_line_height(
            self.font, self.font_size)
        self.line_height, self.line_baseline = tmp

        self.point = self.bounding_box[0]
        return self

    def __next__(self) -> Tuple[Point, Point, int]:
        '''
        Returns evenly spaced regions, with baseline
        '''
        if self.point.y > self.bounding_box[1].y:
            raise StopIteration

        top_l = self.point
        bot_r = Point(self.bounding_box[1].x,
                      self.point.y + self.line_height)

        if bot_r.y > self.bounding_box[1].y:
            raise StopIteration
        self.point += Point(0, self.line_height + self.spacing)
        return (top_l, bot_r, top_l.y + self.line_baseline)


# Precision of space fragmentation
PRECISION = 2


class AreaSpecification:
    '''
    Area specification is used in rules, to specify probability
    of specific rule apply to given row/column
    '''
    def __init__(self, area=0.5, rules: List[Tuple[str, float]] = []):
        '''
        :area: available area to be used by given rule, possible values:
            float, float range, star (all)
        :rules: List of tuples, containing name of rule and it's probability
        to be used
        '''
        if not isinstance(area, (float, str)):
            raise Exception('Area must be floating number or range')

        greedy = False

        # HACK: some better approach
        if isinstance(area, str):
            if re.match(r'^[0-1](,|.)[0-9]+\s*-\s*[0-1](,|.)[0-9]+$', area):
                low, up = map(float, area.split('-'))
                area = (low, up)
            elif area == '*':
                greedy = True
            else:
                area = float(area)

        self.greedy = greedy
        self.area = area
        self.rules: List[Tuple[str, float]] = rules

    @staticmethod
    def from_dict(data) -> 'AreaSpecification':
        '''
        Parses AreaSpecification from dictionary
        '''
        rules = []
        for r in data['rules']:
            p = float(r['probability'])
            n = r['name']
            rules.append((n, p))
        return AreaSpecification(area=data.get('area'), rules=rules)

    def choice_rule(self) -> str:
        '''
        Returns name of rule which has been chosen
        '''
        boundaries = []
        tmp = 0.0
        for r in self.rules:
            tmp += r[1]
            boundaries.append((tmp, r))
        if tmp != 1.0:
            raise Exception('Probabilities has to add up to 1.0')

        r = random.random()
        for b in boundaries:
            if r <= b[0]:
                return b[1]
        raise Exception('Couldn\'t choose rule')

    def choice_area_fraction(self, val: float) -> float:
        '''
        Chooses area based with precision to 2 decimal places
        '''
        if self.greedy:
            return round(val, PRECISION)
        # range of possible values
        if isinstance(self.area, tuple):
            v = round(random.uniform(self.area[0], self.area[1]), PRECISION)
            if v > val:
                raise Exception('No space left!')
            return v
        else:
            if self.area > val:
                raise Exception('No space left!')
            return round(self.area, PRECISION)


class LayoutRegion:
    '''
    LayoutRegion allows to describe how to split given area,
    which rules can be where applied
    '''
    def __init__(self, width=1.0, height=1.0, *, name,
                 padding=0, rows=1, columns=1,
                 content: List[AreaSpecification], **kwargs):
        self.name = name
        self.width = int(width)
        self.height = int(height)
        self.padding = int(padding)
        self.rows = int(rows)
        self.columns = int(columns)
        self.content = content
        self._childs = []  # list containing child elements

    @staticmethod
    def from_dict(data) -> 'LayoutRegion':
        sub_layout_rules = []
        for spec in data['content']:
            sub_layout_rules.append(AreaSpecification.from_dict(spec))

        return LayoutRegion(
            data.get('width', 1.0), data.get('height', 1.0),
            name=data.get('name'),
            padding=data.get('padding', 0), rows=data.get('rows', 1),
            columns=data.get('columns', 1), content=sub_layout_rules
        )

    def deep_copy(self) -> 'LayoutRegion':
        return LayoutRegion(**vars(self))

    def fit_to_region(self, top_l: Point, bot_r: Point):
        '''
        fit_to_region function fits region space to specified rules
        '''
        splitting_axis = None
        split_val = None
        split_cnt = None

        self.bounding_box = (top_l, bot_r)

        if self.columns > 1:
            splitting_axis = 'x'
            split_val = bot_r.x - top_l.x
            split_cnt = self.columns
        elif self.rows > 1:
            splitting_axis = 'y'
            split_val = bot_r.y - top_l.y
            split_cnt = self.rows
        else:
            raise Exception('Invalid LayoutRegion')

        splits = []
        actual_splits = []
        ratio = 1.0
        s0 = 0
        s1 = 0
        for v in range(split_cnt):
            # area = self.content[v].choice_area_fraction(split_val)
            selected_ratio = self.content[v].choice_area_fraction(ratio)
            area = math.floor(selected_ratio * split_val)
            ratio -= selected_ratio
            rule = self.content[v].choice_rule()
            splits.append((area, rule))
            s1 += area
            actual_splits.append((s0, s1))
            s0 = s1

        # call fit to region for each subregion
        for idx in range(split_cnt):
            lefttop = None
            rightbot = None
            s0, s1 = actual_splits[idx]

            if splitting_axis == 'x':
                lefttop = Point(s0, top_l.y)
                rightbot = Point(s1, bot_r.y)
            else:
                lefttop = Point(top_l.x, s0)
                rightbot = Point(bot_r.x, s1)

            rule = get_content(splits[idx][1][0])
            self._childs.append(rule)
            rule.fit_to_region(lefttop, rightbot)

    def __iter__(self):
        iters = []
        for c in self._childs:
            if isinstance(c, LayoutRegion):
                iters.extend(list(c))
            else:
                iters.append(c)
        return iter(iters)


class Region:
    '''
    Region class represent single region region containing lines of text
    or divided into multiple other regions

    allows of iteration through regions returning bounding boxes for specified
    line numbers
    '''
    def __init__(self, width=100.0, height=100.0,
                 *, padding=0, line_height=0, content: List[str],
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


layouts: RegionConfiguration = {}


# XXX: dead function
# def init_layouts(la: Dict[str, RegionConfiguration]):
#     global layouts
#     layouts = la


def get_content(content_name: str) -> RegionConfiguration:
    global layouts
    log.info(f'Layouts content: {layouts}')
    layout = layouts.get(content_name)
    return layout.deep_copy()


def load_layouts(path: str) -> RegionConfiguration:
    '''load_layout loads region layouts specified in file pointed by path'''
    with open(path, 'r') as f:
        d = yaml.safe_load(f)

    regions = RegionConfiguration.from_dict(d)
    global layouts
    layouts = regions.regions
    return regions
