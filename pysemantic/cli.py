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

Options:
    -h --help	Show this screen
"""

from docopt import docopt


def cli(arguments):
    """cli - The main CLI argument parser

    :param arguments: command line arguments
    """
    if arguments.get("list", False):
        from project import view_projects
        view_projects()


def main():
    arguments = docopt(__doc__, version="semantic v0.0.1")
    cli(arguments)
