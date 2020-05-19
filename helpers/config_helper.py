#!/usr/bin/env python3

from configparser import ConfigParser
from collections import defaultdict


def none_factory(): return None


def parse_configuration(config_path):
    config_dict = defaultdict(none_factory)
    config = ConfigParser()
    config.read(config_path)
    for section in config.sections():
        config_dict[section] = _parse_configuration_section(config, section)

    return config_dict


def _parse_configuration_section(config, section):
    section_dict = defaultdict(none_factory)
    options = config.options(section)
    for option in options:
        try:
            value = config.get(section, option)
            int_value = _parse_int(value)
            float_value = _parse_float(value)
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


def _parse_int(s, base=10, value=None):
    try:
        return int(s, base)
    except ValueError:
        return value


def _parse_float(s, value=None):
    if '.' in s:
        try:
            return float(s)
        except ValueError:
            pass
    return value
