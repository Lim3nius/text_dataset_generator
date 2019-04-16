#!/usr/bin/env python
import numpy as np

def addSaltAndPepper(img, prob, s_vs_p_ratio=0.5):
    '''
    <IMPURE> function

    :param img: numpy array reprresenting image, of shape (H,W,C)
    :param prob: float in range [0,1] representing probability of pixel to
      be affected by this
    :param s_vs_p_ratio: salt vs pepper ratio, 0 => everything is salt
      1 => everything is pepper
    :returns: Nothing, img is object, modifications is saved within it
    '''

    pixels = img.shape[0] * img.shape[1]
    salt_cnt = np.ceil(pixels * prob * s_vs_p_ratio)

    # Construct 2 list with randomly selected coordinations for height, width
    coords = [ np.random.randint(0, i - 1, int(salt_cnt)) for i in img.shape[:-1]]
    img[tuple(coords)] = np.array([255,255,255])

    pepper_cnt = np.ceil(pixels * prob * (1 - s_vs_p_ratio))
    coords = [ np.random.randint(0, i - 1, int(pepper_cnt)) for i in img.shape[:-1]]
    img[tuple(coords)] = np.array([0,0,0])
