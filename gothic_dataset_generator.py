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
from helpers import xml_helper


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


def update_annotations(annotations, padding_left, padding_top):
    new_annotations = []
    for annotation in annotations:
        character, position = annotation
        x, y, w, h = position
        new_annotations.append((character, (x+padding_left, y+padding_top, w, h)))

    return new_annotations


def update_baselines(baselines, padding_left, padding_top):
    new_baselines = []
    for baseline in baselines:
        x, y, w = baseline
        new_baselines.append((x + padding_left, y + padding_top, w))

    return new_baselines



def set_paddings(img, config):
    height, width, _ = img.shape
    config['Padding'] = {
        'top': config['Page']['padding'],
        'bottom': config['Page']['padding'],
        'left': config['Page']['padding'],
        'right': config['Page']['padding'],
    }


def modify_line(line, config):
    output = line
    if config["Text"]["tolowercase"]:
        output = output.lower()
    
    if config["Text"]["firstuppercase"] > 0.0:
        prob = config["Text"]["firstuppercase"]
        words = output.split()
        output = ""
        for word in words:
            if np.random.random() < prob:
                output += word.title()
            else:
                output += word

            output += " "

    if config["Text"]["punctuationafterword"]:
        prob = config["Text"]["punctuationafterword"]
        words = output.split()
        output = ""
        punctuations = config["Text"]["punctuations"]
        for word in words:
            output += word
            
            if np.random.random() < prob:
                output += punctuations[np.random.randint(len(punctuations))]
            
            output += " "

    return output.rstrip()


def modify_content(content, config):
    new_content = []
    for line in content:
        new_content.append(modify_line(line, config))

    return new_content


def main():
    args = parse_arguments()
    config = parse_configuration(args.config)

    backgrounds = file_helper.load_all_images(config['Common']['backgrounds'])
    fonts = file_helper.load_all_fonts(config['Common']['fonts'])

    content = file_helper.read_file(config['Common']['input'], config['Text']['words'])
    content = modify_content(content, config)
    
    file_helper.create_directory_if_not_exists(config['Common']['outputs'])

    config["FontSizes"] = {}

    index = 0
    while content:
        background = np.copy(backgrounds[random.randint(0, len(backgrounds) - 1)])
        font = fonts[random.randint(0, len(fonts) - 1)]

        try:
            text_img, annotations, baselines, content = text_renderer.render_page(font, content, config)
            config['Baseline'] = {'text': baselines}
        except Exception as ex:
            traceback.print_exc()
            print("There was an error during creating image number", index)
            print("Font:", font)
            continue

        set_paddings(text_img, config)
        text_img = image_helper.add_padding_to_img(
            text_img,
            padding_top=config['Padding']['top'],
            padding_bottom=config['Padding']['bottom'],
            padding_left=config['Padding']['left'],
            padding_right=config['Padding']['right'])

        annotations = update_annotations(annotations, config['Padding']['left'], config['Padding']['top'])
        baselines = update_baselines(baselines, config['Padding']['left'], config['Padding']['top'])

        try:
            result = effects_helper.apply_effects(text_img, "", {}, font, background, config)
        except Exception as ex:
            traceback.print_exc()
            print("There was an error during applying effects on image number", index)
            print("Font:", font)
            continue

        image_name = "image_" + str(index)

        transkribus = xml_helper.annotations_and_baselines_to_transkribus_xml(annotations, baselines, image_name + ".png", result.shape[:2])
        file_helper.write_file(transkribus, config['Common']['outputs'] + image_name + ".xml")

        file_helper.write_image(result, config['Common']['outputs'] + image_name + ".png")
        file_helper.write_annotation_file(annotations, baselines, config['Common']['outputs'] + image_name + ".txt")

        if config['Common']['annotations']:
            result = image_helper.draw_annotations(result, annotations, baselines)
            file_helper.write_image(result, config['Common']['outputs'] + image_name + "_annotations.png")

        index += 1
        print("Completed " + image_name + ".")
        sys.stdout.flush()

    # file_helper.write_file(output_classes_content, config['Common']['outputs'] + train_or_test + "output.txt")

    return 0


if __name__ == "__main__":
    sys.exit(main())
