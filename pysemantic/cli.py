#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2015 jaidev <jaidev@newton>
#
# Distributed under terms of the MIT license.

"""semantic

Usage:
  semantic list [--project=<PROJECT_NAME>]
  semantic add PROJECT_NAME PROJECT_SPECFILE
  semantic remove PROJECT_NAME [--dataset=<dname>]
  semantic set-schema PROJECT_NAME SCHEMA_FPATH
  semantic set-specs PROJECT_NAME --dataset=<dname> [--path=<pth>] [--dlm=<sep>]
  semantic add-dataset DATASET_NAME --project=<pname> --path=<pth> --dlm=<sep>

Options:
  -h --help	        Show this screen
  -d --dataset=<dname>   Name of the dataset to modify
  --path=<pth>        Path to a dataset
  --dlm=<sep>         Declare the delimiter for a dataset
  -p --project=<pname>   Name of the project to modify

"""

import os.path as op

from docopt import docopt

from pysemantic import project as pr
from pysemantic.errors import MissingProject


def cli(arguments):
    """cli - The main CLI argument parser.

    :param arguments: command line arguments, as parsed by docopt
    :type arguments: dict
    :return: None
    """
    if arguments.get("list", False):
        if arguments['--project'] is None:
            pr.view_projects()
        else:
            proj_name = arguments.get('--project')
            dataset_names = pr.get_datasets(proj_name)
            for name in dataset_names:
                print name
    elif arguments.get("add", False):
        proj_name = arguments.get("PROJECT_NAME")
        proj_spec = arguments.get("PROJECT_SPECFILE")
        proj_spec = op.abspath(proj_spec)
        pr.add_project(proj_name, proj_spec)
    elif arguments.get("remove", False):
        proj_name = arguments.get("PROJECT_NAME")
        if arguments['--dataset'] is None:
            if not pr.remove_project(proj_name):
                print "Removing the project {0} failed.".format(proj_name)
        else:
            pr.remove_dataset(proj_name, arguments['--dataset'])
    elif arguments.get("set-schema", False):
        try:
            proj_name = arguments.get("PROJECT_NAME")
            proj_spec = arguments.get("SCHEMA_FPATH")
            proj_spec = op.abspath(proj_spec)
            pr.set_schema_fpath(proj_name, proj_spec)
        except MissingProject:
            msg = """Project {} not found in the configuration. Please use
            $ semantic add
            to register the project.""".format(arguments.get("PROJECT_NAME"))
            print msg
    elif arguments.get("set-specs", False):
        proj_name = arguments.get("PROJECT_NAME")
        dataset_name = arguments.get("--dataset")
        newspecs = {}
        if arguments.get("--path", False):
            newspecs['path'] = arguments.get("--path")
        if arguments.get("--dlm", False):
            newspecs['delimiter'] = arguments.get("--dlm")
        pr.set_schema_specs(proj_name, dataset_name, **newspecs)
    elif arguments.get("add-dataset", False):
        proj_name = arguments.get('--project')
        dataset_name = arguments.get("DATASET_NAME")
        specs = dict(path=arguments["--path"], delimiter=arguments["--dlm"])
        pr.add_dataset(proj_name, dataset_name, specs)


def main():
    arguments = docopt(__doc__, version="semantic v0.0.1")
    cli(arguments)
