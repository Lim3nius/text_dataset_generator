from __future__ import print_function

import sys
import numpy as np
import random
import math
import signal
import traceback
import copy
import argparse

from freetype import *

from helpers import file_helper
from helpers import image_helper
from helpers import effects_helper
from helpers import text_renderer
from helpers import xml_helper
from helpers import semantic_segmentation_helper
from helpers.config_helper import parse_configuration
from helpers.input_modifications import modify_content
from helpers.postprocess_helper import background_thresholding

def update_annotations(annotations, padding_left, padding_top):
    new_annotations = []
    for annotation in annotations:
        character, position = annotation
        x, y, w, h = position
        x, y, w, h = map(int, [x,y,w,h]) # ensure integer values (float can occur)
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


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-c', '--config', help='Path to the configuration file.', required=True)
    args = parser.parse_args()
    return args


def main():
    args = parse_arguments()
    config = parse_configuration(args.config)

    backgrounds = file_helper.load_all_images(config['Common']['backgrounds'])
    fonts = file_helper.load_all_fonts(config['Common']['fonts'])

    content = file_helper.read_file(config['Common']['input'], config['Text']['words'])
    content = modify_content(content, config)

    file_helper.create_directory_if_not_exists(config['Common']['outputs'])

    config["FontSizes"] = {}
    config["OriginalText"] = copy.deepcopy(content)

    image_names = []
    annotation_names = []
    text_generation_failures = 0 # Counter of unsuccessfull attempts to generate
    # given text

    index = 0
    while content:
        background = np.copy(backgrounds[random.randint(0, len(backgrounds) - 1)])
        font = fonts[random.randint(0, len(fonts) - 1)]

        config["FontSizes"] = {}
        config["Page"]["lineheight"] = np.random.randint(config["Page"]["minlineheight"], config["Page"]["maxlineheight"])

        try:
            text_img, annotations, baselines, new_content = text_renderer.render_page(font, content, config)
            config['Baseline'] = {'text': baselines}
        except Exception as ex:
            traceback.print_exc()

            if text_generation_failures > 5:
                print("Skipping '{}' -- because unable to generate image".format(content[0][:10]))
                content[0] = content[0][10:]
                text_generation_failures = 0
                continue

            print("---", file=sys.stderr)
            print("There was an error during creating image number", index, file=sys.stderr)
            print("Text:", content[0][:30], "...", file=sys.stderr)
            print("Font:", font, file=sys.stderr)
            print("Trying to generate same text again.", file=sys.stderr)
            print("---", file=sys.stderr)
            text_generation_failures += 1
            continue

        text_generation_failures = 0

        set_paddings(text_img, config)
        text_img = image_helper.add_padding_to_img(
            text_img,
            padding_top=config['Padding']['top'],
            padding_bottom=config['Padding']['bottom'],
            padding_left=config['Padding']['left'],
            padding_right=config['Padding']['right'])

        annotations = update_annotations(annotations, config['Padding']['left'], config['Padding']['top'])
        baselines = update_baselines(baselines, config['Padding']['left'], config['Padding']['top'])

        semantic_segmentation_image = None

        if config['Common']['semanticsegmentation']:
            semantic_segmentation_image = semantic_segmentation_helper.generate(text_img, annotations)

        if config['Common']['textgroundtruth']:
            segmented = background_thresholding(text_img)
            file_helper.write_image(segmented, config['Common']['outputs'] + config['Common']['imageprefix'] + '_' + str(index) + '_no_effect.png')

        try:
            result = effects_helper.apply_effects(text_img, font, background, config)
        except Exception as ex:
            traceback.print_exc()
            print("---", file=sys.stderr)
            print("There was an error during applying effects on image number", index, file=sys.stderr)
            print("Text:", content[0][:30], "...", file=sys.stderr)
            print("Font:", font, file=sys.stderr)
            print("Trying to generate same text again.", file=sys.stderr)
            print("---", file=sys.stderr)

            print('Skipping 10 characters. Reason: Unable to generate paragraph')
            content[0]=content[0][10:]
            continue

        content = new_content

        image_name = config['Common']['imageprefix'] + "_" + str(index)

        image_names.append(image_name + ".png")
        annotation_names.append(image_name + ".xml")

        transkribus = xml_helper.annotations_and_baselines_to_transkribus_xml(annotations, baselines, image_name + ".png", result.shape[:2])
        file_helper.write_file(transkribus, config['Common']['outputs'] + image_name + ".xml")

        file_helper.write_image(result, config['Common']['outputs'] + image_name + ".png")

        if config['Common']['semanticsegmentation'] and semantic_segmentation_image is not None:
            file_helper.write_image(semantic_segmentation_image, config['Common']['outputs'] + image_name + "_semantic.png")

        file_helper.write_annotation_file(annotations, baselines, config['Common']['outputs'] + image_name + ".txt")

        if config['Common']['annotations']:
            result = image_helper.draw_annotations(result, annotations, baselines)
            file_helper.write_image(result, config['Common']['outputs'] + image_name + "_annotations.png")

        index += 1
        print("Completed " + image_name + ".")
        sys.stdout.flush()

    file_helper.write_file(xml_helper.mets_transkribus_xml(image_names, annotation_names), config['Common']['outputs'] + "mets.xml")

    # file_helper.write_file(output_classes_content, config['Common']['outputs'] + train_or_test + "output.txt")

    return 0


if __name__ == "__main__":
    ex_code = 0
    try:
        ex_code = main()
    except KeyboardInterrupt:
        print('Generation cancelled')
    sys.exit(ex_code)

