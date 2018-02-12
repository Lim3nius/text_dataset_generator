
import os
import sys
import cv2
import math
import numpy as np


def create_directory_if_not_exists(dir_name):
    import os
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)


def get_word_image(image, left, top, right, bottom, coef, target_height=128, target_width=256):
    height = bottom - top
    width = right - left

    target_height *= coef
    target_width *= coef

    source_top = top - int(math.floor((target_height - height) / 2.))
    source_bottom = bottom + int(math.ceil((target_height - height) / 2.))
    source_left = left - int(math.floor((target_width - width) / 2.))
    source_right = right + int(math.ceil((target_width - width) / 2.))

    return np.copy(image[source_top:source_bottom, source_left:source_right])


def extract_data(directory, output_directory="Images/", target_height=128, target_width=256):
    text_files = [os.path.join(directory, name)
                  for name in os.listdir(directory)
                  if os.path.isfile(os.path.join(directory, name)) and
                  name.endswith(".txt")]

    index = 0
    outputs = []
    first_line = True
    coef = 1

    for file in text_files:
        first_line = True
        image = cv2.imread(file.replace(".txt", ".jpg"))

        with open(file, "r") as f_read:
            for line in f_read:
                if first_line is True:
                    coef = float(line.rstrip())
                    first_line = False

                else:
                    if len(line) > 0:
                        items = line.rstrip().split("\t")

                        left = int(items[0])
                        top = int(items[1])
                        right = int(items[2])
                        bottom = int(items[3])

                        output = items[4]

                        word_image = get_word_image(image, left, top, right, bottom, coef, target_height, target_width)
                        word_image = cv2.resize(word_image, (target_width, target_height))
                        image_path = os.path.join(directory, output_directory) + "image_" + str(index) + ".png"

                        cv2.imwrite(image_path, word_image)

                        outputs.append("image_" + str(index) + ".png\t" + output)

                        index += 1

    with open(os.path.join(directory, output_directory) + "output.txt", "w") as f_write:
        for output in outputs:
            f_write.write(output + "\n")


def main():
    src_directory = "HistoricData/"
    create_directory_if_not_exists(src_directory + "Images/")
    extract_data(src_directory, target_height=96, target_width=256)
    return 0


if __name__ == "__main__":
    sys.exit(main())
