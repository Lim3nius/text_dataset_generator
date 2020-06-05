#!/usr/bin/env python3

"""
File: text_keeper.py
Author: Tomáš Lessy
Email: lessy.mot@gmail.com
Github: https://github.com/lim3nius
Description: This module contains TextKeeper class used as main object for
    text selection and reading data
"""

from typing import List


class TextKeeper:
    '''TextKeeper is class which is used for text manipulation and
    selection'''

    def __init__(self, words: List[str], *, cycle: bool = False):
        self.words = words
        self.cycle = cycle

        # internal variables
        self.pos = 0
        self.last_word = ''

    @staticmethod
    def from_file(path: str, *, cycle: bool = False):
        words = []
        with open(path, 'r') as f:
            for line in f:
                tmp = list(filter(None, line.split(' ')))
                words.extend(tmp)

        return TextKeeper(words, cycle=cycle)

    def get_word(self) -> str:
        '''get_word returns word from text. To continue, one has to
        accept word'''
        if self.pos >= len(self.words):
            if self.cycle:
                self.pos = 0
            else:
                raise ValueError('All text is depleted')

        return self.words[self.pos]

    def accept_word(self):
        self.pos += 1
