#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2015 jaidev <jaidev@newton>
#
# Distributed under terms of the MIT license.

"""
Parser wrappers for data
"""

from validator import DataDictValidator
import pandas as pd
import yaml


def load_from_dictionary(specfile):
    with open(specfile, "r") as f:
        data = yaml.load(f, Loader=yaml.CLoader)
    validators = {}
    outdata = {}
    for k, v in data.iteritems():
        validator = DataDictValidator(name=k, specification=v)
        validators[k] = validator
        outdata[k] = pd.read_table(**validator.get_parser_args())
    return validators, outdata

if __name__ == '__main__':
    validators, datasets = load_from_dictionary("dictionary.yaml")
