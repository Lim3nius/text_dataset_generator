from __future__ import print_function

import sys
from PIL import Image
import numpy as np
import ConfigParser
import random

import file_helper
import image_helper
import text_renderer
import effects_helper

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
    #parser.add_argument('-i', '--input', help='Input text file name.', required=True)
    #parser.add_argument('-o', '--output', help='Output directory name.', required=True)
    #parser.add_argument('-f', '--font', help='Font file name.', required=True)
    #parser.add_argument('-s', '--size', help='Font size.', required=True)
    #parser.add_argument('-b', '--background', help='Background texture.', required=True)
    #parser.add_argument('-a', '--annotations', help='Show annotations.', action='store_true')
    #parser.add_argument('-w', '--words', help='Generate dataset for single words.', action='store_true')
    
    parser.add_argument('-c', '--config', help='Path to the configuration file.', required=True)
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


def main():
    args = parse_arguments()
    config = parse_configuration(args.config)

    backgrounds = file_helper.load_all_images(config['Common']['backgrounds'])
    
    content = file_helper.read_file(config['Common']['input'], config['Common']['words'])
    total = len(content)
    words_dict = build_dict(content)

    file_helper.create_directory_if_not_exists(config['Common']['outputs'])

    output_classes_content = []

    for index, line_original in enumerate(content):
        line = line_original.lower()
        background = np.copy(backgrounds[random.randint(0, len(backgrounds) - 1)])
        text_img, annotations = text_renderer.render_text(config['Common']['font'], line, config['Common']['fontsize'])
        
        text_img = image_helper.add_padding_to_img(text_img, 
                                                   padding_top=config['Padding']['top'],
                                                   padding_bottom=config['Padding']['bottom'],
                                                   padding_left=config['Padding']['left'],
                                                   padding_right=config['Padding']['right'])
        
        result = effects_helper.apply_effects(text_img, line, words_dict, background, config)

        file_helper.write_image(result, config['Common']['outputs'] + "/image_" + str(index) + ".png")
        file_helper.write_annotation_file(annotations, config['Common']['outputs'] + "/image_" + str(index) + ".txt")

        output_classes_content.append("image_" + str(index) + ".png" + "\t" + line)

        if config['Common']['annotations']:
            result = image_helper.draw_annotations(result, annotations, config['Padding']['left'])
            file_helper.write_image(result, config['Common']['outputs'] + "/image_" + str(index) + "_annotations.png")

        print("Completed " + str(index + 1) + "/" + str(total) + ".", end="\r")
        sys.stdout.flush()

    file_helper.write_file(output_classes_content, config['Common']['outputs'] + "/output.txt")
               
    return 0

if __name__ == "__main__":
    sys.exit(main())
