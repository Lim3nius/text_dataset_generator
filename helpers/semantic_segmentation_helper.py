import numpy as np


CHARS=" abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ.,:;-?!0123456789"
THRESHOLD=128


def generate(image, annotations):
    image_slice = image[:, :, 0]
    result = np.zeros(image.shape[:2])
    
    for annotation in annotations:
        character, position = annotation
        x, y, w, h = position

        result[y:y+h, x:x+w] = _convert_char(character, image_slice[y:y+h, x:x+w])

    result = _grayscale_to_rgba(result)

    return result


def _convert_char(character, image):
    if character == " ":
        image = np.zeros(image.shape[:2])

    value = CHARS.find(character) + 1
    if value == 0:
        value = 255

    result = np.where(image < THRESHOLD, value, 0)

    return result


def _grayscale_to_rgba(img):
    w, h = img.shape
    ret = np.empty((w, h, 3), dtype=np.uint8)
    ret[:, :, 0] = img
    ret[:, :, 1] = img
    ret[:, :, 2] = img
    return ret