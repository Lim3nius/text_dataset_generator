from PIL import Image
import random
import numpy as np
import sys
import scipy.misc


def parse_arguments():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--source-path', help='Path to source image.', required=True)
    parser.add_argument('-d', '--destination-path', help='Path to destination image.', required=True)
    args = parser.parse_args()
    return args


def read_image(path):
    img = Image.open(path)
    img = np.array(img)

    if img.shape[2] == 4:
        img = img[:, :, :3]

    return img

    
def write_image(img, path):
    scipy.misc.toimage(img, cmin=0.0, cmax=255.0).save(path)


def get_random_color(min_value=15, max_value=240):
    r = random.randint(min_value, max_value)
    g = random.randint(min_value, max_value)
    b = random.randint(min_value, max_value)
    return [r, g, b]


def colorify(image):
    colored = np.zeros(image.shape)

    colors = {0: [0,0,0]}

    height, width, _ = image.shape

    for y in range(height):
        for x in range(width):
            source_color = image[y, x]
            source_value = source_color[0]

            color = None
            try:
                color = colors[source_value]
            except KeyError:
                color = get_random_color()
                colors[source_value] = color

            colored[y, x] = color

    return colored


def main():
    args = parse_arguments()
    source_image = read_image(args.source_path)
    dest_image = colorify(source_image)
    write_image(dest_image, args.destination_path)

    
if __name__ == "__main__":
    sys.exit(main())
