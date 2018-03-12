from __future__ import print_function

import os
import sys
from PIL import Image
import numpy as np
import cv2
import math

# (HEIGHT, WIDTH)
OUTPUT_SIZE=(48, 64)
PADDING=32


def write_image(img, path):
    im = Image.fromarray(img)
    if im.mode != 'RGB':
        im = im.convert('RGB')

    im.save(path)


def read_image(path):
    img = Image.open(path)
    img = np.array(img)

    if img.shape[2] == 4:
        img = img[:,:,:3]
    
    return img


def add_padding(img, padding=PADDING, value=[0,0,0,0]):
    return cv2.copyMakeBorder(img, padding, padding, padding, padding, cv2.BORDER_CONSTANT, value=value)


def split_tuple(line):
    line = line.rstrip()
    if line.endswith(")"):
        line = line[:-1]

    first_apostroph = line.index("'")
    last_apostroph = line.rindex("'")
    
    character = line[first_apostroph + 1:last_apostroph]
    position = int(line[last_apostroph + 2:])
    
    return character, position

def get_positions(path, output_dir):
    output = []
    with open(path, "r") as f_read:
        content = f_read.readlines()
        for index, line in enumerate(content[:-1]):
            character, position = split_tuple(line)
            _, next_position = split_tuple(content[index + 1])
            output.append((character, position, next_position))

    return output


def extract_subimage(image, position):
    height = image.shape[0]

    start_row = int((height - OUTPUT_SIZE[0]) / 2)
    start_column = int((position - OUTPUT_SIZE[1] / 2) + PADDING)

    return image[start_row:start_row + OUTPUT_SIZE[0], start_column:start_column + OUTPUT_SIZE[1]]


def create(dir1, dir2):
    outputs = []
    dir1 = os.path.realpath(dir1)
    dir2 = os.path.realpath(dir2)
    src_files = [os.path.join(dir1, name) 
                for name in os.listdir(dir1) 
                if os.path.isfile(os.path.join(dir1, name)) and 
                    name.endswith(".txt") and 
                    name != "output.txt"]
    
    for file_path in src_files:
        positions = get_positions(file_path, dir2)
        image_path = file_path.replace(".txt", ".png")
        image = add_padding(read_image(image_path))
        
        for index, (character, position, next_position) in enumerate(positions):
            subimage = extract_subimage(image, position)
            subimage_name = image_path[image_path.rindex("/") + 1:].replace(".png", "_" + str(index) + ".png")
            subimage_path = os.path.join(dir2, subimage_name)
            
            write_image(subimage, subimage_path)

            outputs.append((subimage_name, character, str(position), str(next_position)))

    outputs_path = os.path.join(dir2, "output.txt")
    with open(outputs_path, "w") as f_write:
        for output in outputs:            
            f_write.write("\t".join(output) + "\n")
           

def main():
    src_directory = "Outputs"
    dst_directory = "OutputsPositions"

    create(src_directory + "/train/", dst_directory + "/train/")
    #create(src_directory + "/test/", dst_directory + "/test/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
