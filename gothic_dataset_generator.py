
import sys
from PIL import Image
import numpy as np

import file_helper
import image_helper
import text_renderer

def parse_arguments():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', help='Input text file name.')
    parser.add_argument('-o', '--output', help='Output directory name.')
    parser.add_argument('-f', '--font', help='Font file name.')
    parser.add_argument('-s', '--size', help='Font size.')
    parser.add_argument('-b', '--background', help='Background texture.')
    args = parser.parse_args()
    return args


def main():
    args = parse_arguments()
    content = file_helper.read_file(args.input)
    file_helper.create_directory_if_not_exists(args.output)
        
    for index, line in enumerate(content):
        background = file_helper.read_image(args.background)
        text_img, annotations = text_renderer.render_text(args.font, line, int(args.size))
        result = image_helper.place_text_on_background(text_img, background)
        file_helper.write_image(result, args.output + "/image_" + str(index) + ".png")
        file_helper.write_annotation_file(annotations, args.output + "/image_" + str(index) + ".txt")
                
    return 0;

if __name__ == "__main__":
    sys.exit(main())
