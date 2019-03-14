#!/usr/bin/env python3

import numpy as np

def background_thresholding(arr):
    new_arr = np.array(arr, copy=True)
    assert(len(arr.shape) == 3)

    lower = 0
    higher = 255
    threshold = 127

    l = new_arr[:,:,0] < threshold
    h = new_arr[:,:,0] >= threshold

    new_arr[l,0] = lower
    new_arr[h,0] = higher

    return new_arr

