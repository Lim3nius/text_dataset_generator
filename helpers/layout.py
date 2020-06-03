#!/usr/bin/env python3

import yaml
import math
from logging import getLogger
from sympy import Point
from typing import Tuple, Union, List
import enum
import itertools

log = getLogger()


def load_file(path: str):
    obj = yaml.load(open(path), Loader=yaml.CLoader)
    return obj


class RegionError(Exception):
    pass


directions = ['vertical', 'horizontal']
Direction = enum.IntEnum('Direction', directions)


class Region:
    """docstring for Region"""
    def __init__(self, width=100.0, height=100.0,
                 *, padding=0, line_height=0, sub_regions=None, direction=''):
        if sub_regions and line_height:
            raise RegionError('line_height and sub_regions present')
        elif not (line_height or sub_regions):
            raise RegionError(
                'have to define one of "line_height" or "sub_regions"')

        if sub_regions:
            if direction not in directions:
                raise RegionError(f'Invalid direction value: {direction}')

            if not direction:
                raise RegionError('Unspecified direction for sub regions')

        self.line_height = line_height
        self.width = width
        self.height = height
        self.padding = padding
        self.direction = Direction.vertical if direction == 'vertical' \
            else Direction.horizontal
        self.sub_regions = sub_regions
        self.box = None

    @staticmethod
    def from_string(data: str):
        pass

    def compute_real_val(self, v, dimension):
        if isinstance(v, int):
            return v
        elif isinstance(v, float) and (0 < v <= 1.0):
            return math.floor(dimension * v)
        else:
            raise RegionError('invalid dimension value given')

    def fit_to_region(self, region: Tuple[Point, Point]):
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

        # find splits
        if self.direction is Direction.vertical:
            height_intervals = divide_interval(
                top_left.y, bot_right.y, [r.height for r in self.sub_regions])
            width_intervals = [val(top_left.x, bot_right.x, r.width)
                               for r in self.sub_regions]
        else:
            width_intervals = divide_interval(
                top_left.x, bot_right.x, [r.width for r in self.sub_regions])
            height_intervals = [val(top_left.y, bot_right.y, r.height)
                                for r in self.sub_regions]

        zp = zip(self.sub_regions, width_intervals, height_intervals)
        for (reg, wi, hi) in zp:
            reg.fit_to_region((Point(wi[0], hi[0]), Point(wi[1], hi[1])))

        # slider = top_left
        # for (i, r) in enumerate(self.sub_regions):
        #     if self.direction is Direction.vertical:
        #         height = self.compute_real_val(r.height, reg_height)
        #         new_bot = slider + Point(0, height)
        #         r.fit_to_region(
        #             (slider, Point(bot_right.x, new_bot.y)))
        #         slider = new_bot
        #     else:
        #         width = self.compute_real_val(r.width, reg_width)
        #         new_right = slider + Point(width, 0)
        #         print(f'Fit to region: {new_right}')
        #         r.fit_to_region(
        #             (slider, Point(new_right.x, bot_right.y)))

        #         if i < len(self.sub_regions) - 1:
        #             print(f'Increasing for {i}')
        #             new_right += Point(1, 0)

        #         slider = new_right
        #         print(f'Slider val: {slider}')

    def __iter__(self):
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
            line_iters = [i for i in self.sub_regions.__iter__()]
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
