#!/usr/bin/env python3

import csv

def determine_header_names(config):
    names = ['image']
    for opt in ['annotations', 'semanticsegmentation', 'textgroundtruth']:
        if config['Common'][opt]:
            names.append(opt)
    return names


