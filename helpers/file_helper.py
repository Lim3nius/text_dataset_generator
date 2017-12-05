from PIL import Image
import numpy as np
import string
import os


def read_file(file_name, words=False):
    content = []

    with open(file_name, "r") as f_read:
        for line in f_read:
            if words:
                splitted_words = line.rstrip().split()
                for single_word in splitted_words:
                    content.append(single_word.translate(None, string.punctuation))
            else:
                content.append(line.rstrip())
    
    return content


def write_annotation_file(annotations, file_name):
    with open(file_name, "w") as f_write:
        for annotation in annotations:
            f_write.write(str(annotation) + "\r\n")

def write_file(content, file_name):
    with open(file_name, "w") as f_write:
        for line in content:
            f_write.write(str(line) + "\r\n")


def create_directory_if_not_exists(dir_name):
    import os
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)


def write_image(img, path):
    im = Image.fromarray(img)
    im.save(path)


def read_image(path):
    img = Image.open(path)
    img = np.array(img)

    if img.shape[2] == 4:
        img = img[:,:,:3]
    
    return img


def load_all_images(dir_name):
    images = []
    image_files = [f for f in os.listdir(dir_name) 
                   if os.path.isfile(os.path.join(dir_name, f)) and 
                   (f.endswith(".jpg") or f.endswith(".png"))]

    for file in image_files:
        images.append(read_image(os.path.join(dir_name, file)))

    return images