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
import pprint
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


def add_project(project_name, specfile):
    """ Add a project to the global configuration file.

    :param project_name: Name of the project
    :param specfile: path to the data dictionary used by the project.
    """
    paths = [op.join(os.getcwd(), CONF_FILE_NAME),
             op.join(op.expanduser('~'), CONF_FILE_NAME)]
    for path in paths:
        if op.exists(path):
            parser = RawConfigParser()
            parser.read(path)
            break
    parser.add_section(project_name)
    parser.set(project_name, "specfile", specfile)
    with open(path, "w") as f:
        parser.write(f)


def remove_project(project_name):
    """Remove a project from the global configuration file.

    :param project_name: Name of the project to remove.
    Returns true if the project existed.
    """
    paths = [op.join(os.getcwd(), CONF_FILE_NAME),
             op.join(op.expanduser('~'), CONF_FILE_NAME)]
    for path in paths:
        if op.exists(path):
            parser = RawConfigParser()
            parser.read(path)
            break
    result = parser.remove_section(project_name)
    if result:
        with open(path, "w") as f:
            parser.write(f)
    return result


class Project(object):

    def __init__(self, project_name, parser=pd.read_table):
        """__init__

        :param project_name:
        :param parser:
        """
        self.project_name = project_name
        self.specfile = _get_default_specfile(self.project_name)
        self.validators = {}
        self.parser = parser
        with open(self.specfile, 'r') as f:
            specifications = yaml.load(f, Loader=yaml.CLoader)
        for name, specs in specifications.iteritems():
            self.validators[name] = DataDictValidator(specification=specs,
                                                      specfile=self.specfile,
                                                      name=name)

    def get_dataset_specs(self, dataset_name=None):
        """get_dataset_specs

        :param dataset_name:
        """
        if dataset_name is not None:
            return self.validators[dataset_name].get_parser_args()
        else:
            specs = {}
            for name, validator in self.validators.iteritems():
                specs[name] = validator.get_parser_args()
            return specs

    def view_dataset_specs(self, dataset_name=None):
        """view_dataset_specs

        :param dataset_name:
        """
        specs = self.get_dataset_specs(dataset_name)
        pprint.pprint(specs)

    def set_dataset_specs(self, dataset_name, specs, write_to_file=False):
        """set_dataset_specs

        :param dataset_name:
        :param specs:
        """
        validator = self.validators[dataset_name]
        return validator.set_parser_args(specs, write_to_file)

    def load_dataset(self, dataset_name):
        """load_dataset

        :param dataset_name:
        """
        validator = self.validators[dataset_name]
        return self.parser(**validator.get_parser_args())

    def load_datasets(self):
        """load_datasets"""
        datasets = {}
        for name in self.validators.iterkeys():
            datasets[name] = self.load_dataset(name)
        return datasets


if __name__ == '__main__':
    specfile = _get_default_specfile("valuefirst")
