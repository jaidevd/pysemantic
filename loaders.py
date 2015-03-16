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

CONF_FILE_NAME = "pysemantic.conf"


def _get_default_data_dictionary(project_name, config_fname=CONF_FILE_NAME):
    """_get_default_data_dictionary

    Returns the specifications file used by the given project. The
    configuration file is searched for first in the current directory and then
    in the home directory.

    :param project_name: Name of the project for which to get the spcfile.
    """
    paths = [op.join(os.getcwd(), config_fname),
             op.join(op.expanduser('~'), config_fname)]
    for path in paths:
        if op.exists(path):
            parser = RawConfigParser()
            parser.read(path)
            return parser.get(project_name, 'specfile')

if __name__ == '__main__':
    specfile = _get_default_data_dictionary("valuefirst")
