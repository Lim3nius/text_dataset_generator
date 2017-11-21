from __future__ import print_function

import os
import sys
import numpy as np
from PIL import Image


def build_dict(directory):
    counter = 0
    word_dict = {}

    content_train = read_file(directory + "train/output.txt")
    content_test = read_file(directory + "test/output.txt")

    for line in content_train:
        _, output_class = line.split("\t")
        if output_class not in word_dict.keys():
            word_dict[output_class] = counter
            counter += 1

    for line in content_test:
        _, output_class = line.split("\t")
        if output_class not in word_dict.keys():
            word_dict[output_class] = counter
            counter += 1

    return word_dict


def read_file(path):
    content = []
    with open(path, "r") as f_read:
        for line in f_read:
            content.append(line.rstrip())

    return content


def read_image(path):
    img = Image.open(path)
    img = np.array(img)

    if img.shape[2] == 4:
        img = img[:, :, :3]

    return img


def read_subdir(subdir, word_dict):
    content = read_file(subdir + "output.txt")

    images = []
    classes = []

    for line in content:
        file_name, output_class = line.split("\t")
        images.append(read_image(subdir + file_name))
        classes.append(word_dict[output_class])

    return images, classes


def read(directory="../Outputs/"):
    word_dict = build_dict(directory)
    train_images, train_labels = read_subdir(directory + "train/", word_dict)
    test_images, test_labels = read_subdir(directory + "test/", word_dict)

    return train_images, train_labels, test_images, test_labels


def main():
    train_images, train_labels, test_images, test_labels = read()
    return 0


if __name__ == "__main__":
    sys.exit(main())
