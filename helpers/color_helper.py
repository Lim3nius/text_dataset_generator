#!/usr/bin/env python

import numpy as np
import random

class PreColors:
    RED=np.array([255,0,0],dtype=np.uint8)
    BLUE=np.array([0,255,0],dtype=np.uint8)
    GREEN=np.array([0,0,255],dtype=np.uint8)
    BLACK=np.array([0,0,0],dtype=np.uint8)
    WHITE=np.array([255,255,255],dtype=np.uint8)

def _replaceColorInImg(img, prevCol, newCol):
    '''
    :param img: numpy array (h,w,channels==4)
    :param prevCol: numpy array representing RGB color previous
    :param newCol: numpy array representing RGB color replacing previous
    :return: new img with replaced colors
    '''
    assert(len(img.shape) == 3)
    assert(img.shape[-1] == 4)

    res = np.copy(img)
    shape = img.shape
    res = np.reshape(res, [-1,4])

    for i in range(len(res)):
        if np.array_equal(res[i][:3], prevCol):
            res[i] = np.array([list(newCol) + [res[i,-1]]])

    return np.reshape(res, shape)

def text_color_handle(img, conf):
    '''
    Function which changes text color according to option
    :param img: numpy array representing image
    :param conf: configuration loaded
    :return: img with right color
    '''

    opt = conf['Text']['color'].lower()
    if opt == 'black':
        return img
    elif opt == 'red':
        return _replaceColorInImg(img, PreColors.BLACK, PreColors.RED)
    elif opt == 'blue':
        return _replaceColorInImg(img, PreColors.BLACK, PreColors.BLUE)
    elif opt == 'green':
        return _replaceColorInImg(img, PreColors.BLACK, PreColors.GREEN)
    elif opt == 'random':
        color = random.choice([PreColors.RED, PreColors.BLACK, PreColors.BLUE, PreColors.GREEN])
        return _replaceColorInImg(img, PreColors.BLACK, color)
