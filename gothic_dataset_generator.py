from __future__ import print_function

import sys
from PIL import Image
import numpy as np
import ConfigParser
import random
import math
import signal
import traceback

from freetype import *

from helpers import file_helper
from helpers import image_helper
from helpers import effects_helper
from helpers import text_renderer


def parse_configuration(config_path):
    config_dict = {}
    config = ConfigParser.ConfigParser()
    config.read(config_path)
    for section in config.sections():
        config_dict[section] = parse_configuration_section(config, section)

    return config_dict


def parse_configuration_section(config, section):
    section_dict = {}
    options = config.options(section)
    for option in options:
        try:
            value = config.get(section, option)
            int_value = parse_int(value)
            float_value = parse_float(value)
            if value == 'True':
                section_dict[option] = True
            elif value == 'False':
                section_dict[option] = False
            elif float_value is not None:
                section_dict[option] = float_value
            elif int_value is not None:
                section_dict[option] = int_value
            else:
                section_dict[option] = value
        except:
            print("exception on %s!" % option)
            section_dict[option] = None
    return section_dict


def parse_int(s, base=10, value=None):
    try:
        return int(s, base)
    except ValueError:
        return value


def parse_float(s, value=None):
    if '.' in s:
        try:
            return float(s)
        except ValueError:
            pass
    return value


def parse_arguments():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-c', '--config', help='Path to the configuration file.', required=True)
    args = parser.parse_args()
    return args


def build_dict(content):
    words_dict = {}
    for word in content:
        try:
            words_dict[word] += 1
        except KeyError:
            words_dict[word] = 1

    return words_dict


def update_annotations(annotations, padding):
    new_annotations = []
    for annotation in annotations:
        new_annotations.append((annotation[0], annotation[1] + padding))

    return new_annotations


def set_paddings(img, config):
    height, width, _ = img.shape
    config['Padding'] = {
        'top': config['OutputSize']['padding'],
        'bottom': config['OutputSize']['padding'],
        'left': config['OutputSize']['padding'],
        'right': config['OutputSize']['padding'],
    }


def set_width_and_height(img, config):
    height, width, _ = img.shape
    config['OutputSize']['width'] = width
    config['OutputSize']['height'] = height


def main():
    args = parse_arguments()
    config = parse_configuration(args.config)

    backgrounds = file_helper.load_all_images(config['Common']['backgrounds'])
    fonts = file_helper.load_all_fonts(config['Common']['fonts'])

    content = file_helper.read_file(
        config['Common']['input'], config['Common']['words'])
    total = len(content)
    words_dict = build_dict(content)

    file_helper.create_directory_if_not_exists(config['Common']['outputs'])
    file_helper.create_directory_if_not_exists(
        config['Common']['outputs'] + "train/")
    file_helper.create_directory_if_not_exists(
        config['Common']['outputs'] + "test/")

    config["FontSizes"] = {}

    train_or_test = "train/"

    output_classes_content = []

    for index, line_original in enumerate(content):
        if train_or_test == "train/" and float(index) / float(total) > config['Common']['trainratio']:
            file_helper.write_file(
                output_classes_content, config['Common']['outputs'] + train_or_test + "output.txt")
            train_or_test = "test/"
            output_classes_content = []

        line = line_original.lower()

        background = np.copy(
            backgrounds[random.randint(0, len(backgrounds) - 1)])
        font = fonts[random.randint(0, len(fonts) - 1)]

        config["SurroundingText"]["linespace"] = random.randint(
            config["SurroundingText"]["minlinespace"], 
            config["SurroundingText"]["maxlinespace"])

        config["SurroundingText"]["wordspace"] = random.randint(
            config["SurroundingText"]["minwordspace"], 
            config["SurroundingText"]["maxwordspace"])

        try:
            text_img, annotations, baseline = text_renderer.render_text(
                font, line, config)
            config['Baseline'] = {'text': baseline}
        except Exception as ex:
            print(ex)
            print("There was an error during creating image number", index)
            print("Text:", line)
            print("Font:", font)
            continue

        set_paddings(text_img, config)
        text_img = image_helper.add_padding_to_img(
            text_img,
            padding_top=config['Padding']['top'],
            padding_bottom=config['Padding']['bottom'],
            padding_left=config['Padding']['left'],
            padding_right=config['Padding']['right'])

        set_width_and_height(text_img, config)
        annotations = update_annotations(
            annotations, config['Padding']['left'])

        try:
            result = effects_helper.apply_effects(
                text_img, line, words_dict, font, background, config)
        except Exception as ex:
            print(ex)
            traceback.print_exc()
            print("There was an error during applying effects on image number", index)
            print("Text:", line)
            print("Font:", font)
            continue

        file_helper.write_image(
            result, config['Common']['outputs'] + train_or_test + "image_" + str(index) + ".png")
        file_helper.write_annotation_file(
            annotations, config['Common']['outputs'] + train_or_test + "image_" + str(index) + ".txt")

        output_classes_content.append(
            "image_" + str(index) + ".png" + "\t" + line)

        if config['Common']['annotations']:
            result = image_helper.draw_annotations(result, annotations)
            file_helper.write_image(
                result, config['Common']['outputs'] + train_or_test + "image_" + str(index) + "_annotations.png")

        print("Completed " + str(index + 1) + "/" + str(total) + ".")
        sys.stdout.flush()

    file_helper.write_file(
        output_classes_content, config['Common']['outputs'] + train_or_test + "output.txt")

    return 0


if __name__ == "__main__":
    sys.exit(main())
