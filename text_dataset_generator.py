#!/usr/bin/env python3

import sys
import traceback
import copy
import argparse
from logging import getLogger
from signal import signal, SIGINT, SIGTERM
from multiprocessing import Process
import numpy as np

from helpers import file_helper
from helpers import image_helper
from helpers import effects_helper
from helpers import text_renderer
from helpers import xml_helper
from helpers import semantic_segmentation_helper
from helpers.config_helper import parse_configuration
from helpers.input_modifications import modify_content
from helpers.postprocess_helper import background_thresholding
from helpers import manifest_helper
from helpers import color_helper
from helpers.dict_writer import SyncWriterWrapper
from helpers import baseline_helper
from helpers import logger

log = getLogger()


def update_annotations(annotations, padding_left, padding_top):
    new_annotations = []
    for annotation in annotations:
        character, position = annotation
        x, y, w, h = position
        # ensure integer values (float can occur)
        x, y, w, h = map(int, [x, y, w, h])
        new_annotations.append(
            (character, (x+padding_left, y+padding_top, w, h)))

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
    parser.add_argument('--config', default='config.ini',
                        help='Path to the configuration file. If not provided'
                        '"config.ini" in working directory will be used')
    parser.add_argument('-c', '--count', type=int, default=10**6,
                        help='Number of images to generate')
    parser.add_argument('-w', '--workers', default=1, type=int,
                        help="Number of paralell workers to start")
    parser.add_argument("-l", "--log-level", default='info', type=str,
                        choices=['debug', 'info', 'warning', 'error'],
                        help="Level of logging")
    parser.add_argument("--prefix", default=None, type=str,
                        help=("Prefix for newly generated images."
                              "Config has higer priority"))

    sp = parser.add_subparsers(help="sub command")
    pb = sp.add_parser("baseline", help="foo")
    pb.add_argument('path', help="Path to file with baseline info")
    pb.set_defaults(func=baseline_helper.main)

    args = parser.parse_args()
    return args


def generator(config, content, index, fonts, backgrounds, args,
              manifest_wrtr, start):
    # Counter of unsuccessfull attempts to generate given text
    text_generation_failures = 0
    manifest_row = {}
    index += start
    generated = 0
    to_generate = args.count // args.workers
    step = args.workers
    prefix = args.prefix if args.prefix else config['Common']['imageprefix']

    # if job can't be perfectly divided, give remaining to first worker
    rest = args.count % args.workers
    if rest > 0 and start == 0:
        to_generate += rest

    while content and generated < to_generate:
        background_name, background = np.copy(backgrounds.random_pair())
        font_name, face = fonts.random_pair()

        config["FontSizes"] = {}
        config["Page"]["lineheight"] = np.random.randint(
            config["Page"]["minlineheight"], config["Page"]["maxlineheight"])

        try:
            text_img, annotations, baselines, new_content = text_renderer.render_page(face, content, config)
            config['Baseline'] = {'text': baselines}
        except Exception:
            traceback.print_exc()

            if text_generation_failures > 5:
                print("Skipping '{}' -- because unable to generate image".
                      format(content[0][:10]))
                content[0] = content[0][10:]
                text_generation_failures = 0
                continue

            print("---", file=sys.stderr)
            print("There was an error during creating image number",
                  index, file=sys.stderr)
            print("Text:", content[0][:30], "...", file=sys.stderr)
            print("Font:", font_name, file=sys.stderr)
            print("Trying to generate same text again.", file=sys.stderr)
            print("---", file=sys.stderr)
            text_generation_failures += 1
            continue

        text_generation_failures = 0
        text_img = color_helper.text_color_handle(text_img, config)

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
        image_name = prefix + "_" + str(index)

        if config['Common']['semanticsegmentation']:
            semantic_segmentation_image = semantic_segmentation_helper.generate(text_img, annotations)

        if config['Common']['textgroundtruth']:
            segmented = background_thresholding(text_img)
            file_helper.write_image(segmented, config['Common']['outputs'] +
                                     image_name + '_no_effect.png')
            manifest_row['textgroundtruth'] = image_name + '_no_effect.png'

        try:
            result = effects_helper.apply_effects(text_img, face, background,
                                                  config)
        except Exception:
            traceback.print_exc()
            print("---", file=sys.stderr)
            print("There was an error during applying effects on image number", (index), file=sys.stderr)
            print("Text:", content[0][:30], "...", file=sys.stderr)
            print("Font:", font_name, file=sys.stderr)
            print("Trying to generate same text again.", file=sys.stderr)
            print("---", file=sys.stderr)

            print('Skipping 10 characters. Reason: Unable to generate paragraph')
            content[0] = content[0][10:]
            continue

        content = new_content

        manifest_row['image'] = image_name + '.png'
        manifest_row['font'] = font_name
        path = config['Common']['outputs'] + image_name

        # image_names.append(image_name + ".png")
        # annotation_names.append(image_name + ".xml")

        transkribus = xml_helper.annotations_and_baselines_to_transkribus_xml(annotations, baselines, image_name + ".png", result.shape[:2])
        file_helper.write_file(transkribus, path + ".xml")

        file_helper.write_image(result, path + ".png")

        if config['Common']['semanticsegmentation'] and semantic_segmentation_image is not None:
            file_helper.write_image(semantic_segmentation_image, path + "_semantic.png")
            manifest_row['semanticsegmentation'] = image_name + '_semantic.png'

        file_helper.write_annotation_file(annotations, baselines, path + ".txt")

        if config['Common']['annotations']:
            result = image_helper.draw_annotations(result, annotations, baselines)
            file_helper.write_image(result, path + "_annotations.png")
            manifest_row['semanticsegmentation'] = image_name + '_annotations.png'

        index += step
        generated += 1
        manifest_wrtr.writerow(manifest_row)
        print("Completed " + image_name + ".")
        sys.stdout.flush()

    print(f'Worker {start} exited')


class Storage:
    """docstring for ClassName"""
    def __init__(self, config):
        self.config = config

        self.backgrounds = file_helper.load_images(
            config['Common']['backgrounds'])
        self.fonts = file_helper.load_fonts(config['Common']['fonts'])


def main():
    args = parse_arguments()
    logger.setup_logger(args.log_level)
    log.info('Generator started')

    config = parse_configuration(args.config)
    log.info('Configuration parsed')
    prefix = args.prefix if args.prefix else config['Common']['imageprefix']
    storage = Storage(config)
    log.debug('Storage loaded')

    backgrounds = storage.backgrounds
    fonts = storage.fonts

    try:
        args.func(args.path, config, storage)
    except Exception:
        pass

    content = file_helper.read_file(config['Common']['input'],
                                    config['Text']['words'])
    content = modify_content(content, config)

    file_helper.create_directory_if_not_exists(config['Common']['outputs'])

    config["FontSizes"] = {}
    config["OriginalText"] = copy.deepcopy(content)

    # TODO: add back
    # image_names = []
    # annotation_names = []

    # field_names = manifest_helper.determine_header_names(config)
    manifest_wrtr = SyncWriterWrapper(config['Common']['outputs'] +
                                      '/' + prefix + '_manifest.csv')

    index = config['Common']['numberstart']
    pcs = []
    log.info(f'starting {args.workers} workers')

    # proper handling of Ctl-c
    def teardown(signum, frame):
        log.debug('Terminating processes')
        for p in pcs:
            p.kill()

    signal(SIGINT, teardown)
    signal(SIGTERM, teardown)

    try:
        for i in range(args.workers):
            pcs.append(Process(target=generator,
                               args=(config, copy.deepcopy(content), index,
                                     copy.deepcopy(fonts), backgrounds, args,
                                     manifest_wrtr, i)))
            pcs[-1].start()

        log.debug('All workers started')

        for i in range(args.workers):
            pcs[i].join()

        # manifest_wrtr.close()
    except Exception as ex:
        print(f'Something went wrong: {ex}')
        print(traceback.format_exc())
        raise ex

    manifest_wrtr.close()
    return 0


if __name__ == "__main__":
    ex_code = 0
    try:
        ex_code = main()
    except KeyboardInterrupt:
        log.info('Stopped by user')

    except Exception as e:
        log.warning(f'Stopped because of exception: {e}')

    sys.exit(ex_code)
