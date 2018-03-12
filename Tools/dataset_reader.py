from __future__ import print_function

import os
import sys
import cv2
import numpy as np
from PIL import Image


def build_dict(directory):
    counter = 0
    word_dict = {}

    content_train = read_file(directory + "train/output.txt")
    content_test = read_file(directory + "test/output.txt")

    for line in content_train:
        if len(line) > 0:
            _, output_class = line.split("\t")
            if output_class not in word_dict.keys():
                word_dict[output_class] = counter
                counter += 1

    for line in content_test:
        if len(line) > 0:
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
        if len(line) > 0:
            file_name, output_class = line.split("\t")
            images.append(read_image(subdir + file_name))
            classes.append(word_dict[output_class])

    return images, classes


def read_subdir_positions(subdir):
    content = read_file(subdir + "output.txt")

    images = []
    characters = []
    positions = []

    for line in content:
        if len(line) > 0:
            file_name, character, current_position, next_position = line.split("\t")
            images.append(read_image(subdir + file_name))
            characters.append(character)
            positions.append(int(next_position) - int(current_position))

    return images, characters, positions


def read_word_classification(directory="../Outputs/"):
    word_dict = build_dict(directory)
    train_images, train_labels = read_subdir(directory + "train/", word_dict)
    test_images, test_labels = read_subdir(directory + "test/", word_dict)

    train_images = np.asarray(train_images)
    train_labels = np.asarray(train_labels)
    test_images = np.asarray(test_images)
    test_labels = np.asarray(test_labels)

    return train_images, train_labels, test_images, test_labels


def read_character_position(directory="../OutputsPositions/"):
    train_images, train_chars, train_deltas = read_subdir_positions(directory + "train/")
    test_images, test_chars, test_deltas = read_subdir_positions(directory + "test/")

    train_images = np.asarray(train_images)
    train_chars = np.asarray(train_chars)
    train_deltas = np.asarray(train_deltas)
    test_images = np.asarray(test_images)
    test_chars = np.asarray(test_chars)
    test_deltas = np.asarray(test_deltas)

    return train_images, train_chars, train_deltas, test_images, test_chars, test_deltas


def main():
    train_images, train_chars, train_deltas, test_images, test_chars, test_deltas = read_character_position()

    print("Train images: ", train_images.shape)
    print("Train chars: ", train_chars.shape)
    print("Train deltas: ", train_deltas.shape)
    print("Test images: ", test_images.shape)
    print("Test chars: ", test_chars.shape)
    print("Test deltas: ", test_deltas.shape)

    return 0


if __name__ == "__main__":
    sys.exit(main())
