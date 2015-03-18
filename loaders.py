#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2015 jaidev <jaidev@newton>
#
# Distributed under terms of the MIT license.

"""

"""

import os.path as op
import os
from ConfigParser import RawConfigParser
from validator import DataDictValidator
import yaml
import pandas as pd

CONF_FILE_NAME = "pysemantic.conf"


def _get_default_specfile(project_name):
    """_get_default_data_dictionary

    Returns the specifications file used by the given project. The
    configuration file is searched for first in the current directory and then
    in the home directory.

    :param project_name: Name of the project for which to get the spcfile.
    """
    paths = [op.join(os.getcwd(), CONF_FILE_NAME),
             op.join(op.expanduser('~'), CONF_FILE_NAME)]
    for path in paths:
        if op.exists(path):
            parser = RawConfigParser()
            parser.read(path)
            return parser.get(project_name, 'specfile')


class Project(object):

    def __init__(self, project_name, parser=pd.read_table):
        self.project_name = project_name
        self.specfile = _get_default_specfile(self.project_name)
        self.validators = {}
        self.parser = parser
        with open(self.specfile, 'r') as f:
            specifications = yaml.load(f, Loader=yaml.CLoader)
        for name, specs in specifications.iteritems():
            self.validators[name] = DataDictValidator(specification=specs)

    def load_dataset(self, dataset_name):
        validator = self.validators[dataset_name]
        return self.parser(**validator.get_parser_args())

    def load_datasets(self):
        datasets = {}
        for name in self.validators.iterkeyes():
            datasets[name] = self.load_dataset(name)
        return datasets


if __name__ == '__main__':
    specfile = _get_default_specfile("valuefirst")
