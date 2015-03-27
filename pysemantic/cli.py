#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2015 jaidev <jaidev@newton>
#
# Distributed under terms of the MIT license.

"""semantic

Usage:
    semantic list
    semantic add PROJECT_NAME PROJECT_SPECFILE
    semantic remove PROJECT_NAME

Options:
    -h --help	Show this screen
"""

from docopt import docopt
import project as pr


def cli(arguments):
    """cli - The main CLI argument parser

    :param arguments: command line arguments
    """
    if arguments.get("list", False):
        pr.view_projects()
    elif arguments.get("add", False):
        proj_name = arguments.get("PROJECT_NAME")
        proj_spec = arguments.get("PROJECT_SPECFILE")
        pr.add_project(proj_name, proj_spec)
    elif arguments.get("remove", False):
        proj_name = arguments.get("PROJECT_NAME")
        if not pr.remove_project(proj_name):
            print "Removing the project failed."


def main():
    arguments = docopt(__doc__, version="semantic v0.0.1")
    cli(arguments)
