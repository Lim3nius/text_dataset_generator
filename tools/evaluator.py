from __future__ import print_function

import sys

import numpy as np

# from keras.models import load_model

HEIGHT = 48
WIDTH = 64

def parse_arguments():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--position-net', help='Path to position net.', required=True)
    parser.add_argument('-c', '--character-net', help='Path to character net.', required=True)
    parser.add_argument('-d', '--dataset', help='Path to dataset file.', required=True)
    args = parser.parse_args()
    return args


def get_subimage(image, position, size):
    image_height, image_width, _ = image.shape
    width, height = size

    top = (image_height - height) / 2
    bottom = top + height
    left = position - width / 2
    right = left + width
    return image[top:bottom, left:right]


def get_char(character_net, image):
    image = np.expand_dims(image, axis=0)
    probs = character_net.predict(image)
    return probs


def get_position(position_net, image, classified_char):
    image = np.expand_dims(image, axis=0)
    classified_char = classified_char - ord('a')
    chars = np.zeros(26)
    chars[classified_char] = 1.
    probs = position_net.predict(x={'image_input': image, 'character_input': chars})
    return probs


def transcript_image(position_net, character_net, image, first_char_position, transcription_length):
    transcription = ""
    actual_position = first_char_position
    for i in range(transcription_length):
        sub_image = get_subimage(image, actual_position, (WIDTH, HEIGHT))

        char_probs = get_char(character_net, sub_image)
        print(char_probs)
        sys.exit()
        
        
        
        classified_char = np.random.choice(char_probs)

        actual_position += get_position(position_net, sub_image, classified_char)

        transcription += classified_char
    
    return transcription


def compare_transcriptions(original, evaluated):
    correct = 0
    for index, character in enumerate(original):
        if character == evaluated[index]:
            correct += 1

    return float(correct) / len(original)


def evaluate_nets(position_net, character_net, dataset):
    images, transcriptions, first_char_positions = dataset
    accuracies = []

    for i in range(len(images)):
        image = image[i]
        original_transcription = transcriptions[i]
        first_char_position = first_char_positions[i]
        
        evaluated_transcription = transcript_image(position_net, character_net, image, first_char_position, len(original_transcription))
        accuracies.append(compare_transcriptions(original_transcription, evaluated_transcription))

    return accuracies


def load_dataset(file_path):
    images = []
    transcriptions = []
    first_char_positions = []

    with open(file_path, "r") as f_read:
        for line in f_read:
            image_path, transcription, first_char_position = line.rstrip().split("\t")
            images.append(read_image(image_path))
            transcriptions.append(transcription)
            first_char_positions.append(int(first_char_position))
    
    return (images, transcriptions, first_char_positions)


def main():
    args = parse_arguments()
    
    position_net = load_model(args.position_net)
    character_net = load_model(args.character_net)

    dataset = load_dataset(args.dataset)

    accuracies = evaluate_nets(position_net, character_net, dataset)

    return 0


if __name__ == "__main__":
    sys.exit(main())