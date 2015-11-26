#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2015 jaidev <jaidev@newton>
#
# Distributed under terms of the BSD 3-clause license.

"""The Project class."""

import os
import textwrap
import pprint
import logging
import json
from ConfigParser import RawConfigParser
import os.path as op

import yaml
import pandas as pd
import numpy as np

from pysemantic.validator import SchemaValidator, DataFrameValidator, ParseErrorHandler
from pysemantic.errors import MissingProject, MissingConfigError, ParserArgumentError
from pysemantic.loggers import setup_logging
from pysemantic.utils import TypeEncoder
from pysemantic.exporters import AerospikeExporter

try:
    from yaml import CDumper as Dumper
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Dumper
    from yaml import Loader

CONF_FILE_NAME = os.environ.get("PYSEMANTIC_CONFIG", "pysemantic.conf")
logger = logging.getLogger(__name__)


def locate_config_file():
    """Locates the configuration file used by semantic.

    :return: Path of the pysemantic config file.
    :rtype: str
    :Example:

    >>> locate_config_file()
    '/home/username/pysemantic.conf'
    """
    paths = [op.join(os.getcwd(), CONF_FILE_NAME),
             op.join(op.expanduser('~'), CONF_FILE_NAME)]
    for path in paths:
        if op.exists(path):
            logger.info("Config file found at {0}".format(path))
            return path
    raise MissingConfigError("No pysemantic configuration file was fount at"
                             " {0} or {1}".format(*paths))


def get_default_specfile(project_name):
    """Returns the specifications file used by the given project. The \
            configuration file is searched for first in the current directory \
            and then in the home directory.

    :param project_name: Name of the project for which to get the spcfile.
    :type project_name: str
    :return: Path to the data dictionary of the project.
    :rtype: str
    :Example:

    >>> get_default_specfile('skynet')
    '/home/username/projects/skynet/schema.yaml'
    """
    path = locate_config_file()
    parser = RawConfigParser()
    parser.read(path)
    return parser.get(project_name, 'specfile')


def add_project(project_name, specfile):
    """Add a project to the global configuration file.

    :param project_name: Name of the project
    :param specfile: path to the data dictionary used by the project.
    :type project_name: str
    :type specfile: str
    :return: None
    """
    path = locate_config_file()
    parser = RawConfigParser()
    parser.read(path)
    parser.add_section(project_name)
    parser.set(project_name, "specfile", specfile)
    with open(path, "w") as f:
        parser.write(f)


def add_dataset(project_name, dataset_name, dataset_specs):
    """Add a dataset to a project.

    :param project_name: Name of the project to which the dataset is to be \
            added.
    :param dataset_name: Name of the dataset to be added.
    :param dataset_specs: Specifications of the dataset.
    :type project_name: str
    :type dataset_name: str
    :type dataset_specs: dict
    :return: None
    """
    data_dict = get_default_specfile(project_name)
    with open(data_dict, "r") as f:
        spec = yaml.load(f, Loader=Loader)
    spec[dataset_name] = dataset_specs
    with open(data_dict, "w") as f:
        yaml.dump(spec, f, Dumper=Dumper, default_flow_style=False)


def remove_dataset(project_name, dataset_name):
    """Removes a dataset from a project.

    :param project_name: Name of the project
    :param dataset_name: Name of the dataset to remove
    :type project_name: str
    :type dataset_name: str
    :return: None
    """
    data_dict = get_default_specfile(project_name)
    with open(data_dict, "r") as f:
        spec = yaml.load(f, Loader=Loader)
    del spec[dataset_name]
    with open(data_dict, "w") as f:
        yaml.dump(spec, f, Dumper=Dumper, default_flow_style=False)


def get_datasets(project_name=None):
    """Get names of all datasets registered under the project `project_name`.

    :param project_name: name of the projects to list the datasets from. If \
            `None` (default), datasets under all projects are returned.
    :type project_name: str
    :return: List of datasets listed under `project_name`, or if \
            `project_name` is `None`, returns dictionary such that \
            {project_name: [list of projects]}
    :rtype: dict or list
    :Example:

    >>> get_datasets('skynet')
    ['sarah_connor', 'john_connor', 'kyle_reese']
    >>> get_datasets()
    {'skynet': ['sarah_connor', 'john_connor', 'kyle_reese'],
     'south park': ['stan', 'kyle', 'cartman', 'kenny']}
    """
    if project_name is not None:
        specs = get_schema_specs(project_name)
        return specs.keys()
    else:
        dataset_names = {}
        projects = get_projects()
        for project_name, _ in projects:
            dataset_names[project_name] = get_datasets(project_name)
        return dataset_names


def set_schema_fpath(project_name, schema_fpath):
    """Set the schema path for a given project.

    :param project_name: Name of the project
    :param schema_fpath: path to the yaml file to be used as the schema for \
            the project.
    :type project_name: str
    :type schema_fpath: str
    :return: True, if setting the schema path was successful.
    :Example:

    >>> set_schema_fpath('skynet', '/path/to/new/schema.yaml')
    True
    """
    path = locate_config_file()
    parser = RawConfigParser()
    parser.read(path)
    if project_name in parser.sections():
        if not parser.remove_option(project_name, "specfile"):
            raise MissingProject
        else:
            parser.set(project_name, "specfile", schema_fpath)
            with open(path, "w") as f:
                parser.write(f)
            return True
    raise MissingProject


def get_projects():
    """Get the list of projects currently registered with pysemantic as a
    list.

    :return: List of tuples, such that each tuple is (project_name, \
            location_of_specfile)
    :rtype: list
    :Example:

    >>> get_projects()
    ['skynet', 'south park']
    """
    path = locate_config_file()
    parser = RawConfigParser()
    parser.read(path)
    projects = []
    for section in parser.sections():
        project_name = section
        specfile = parser.get(section, "specfile")
        projects.append((project_name, specfile))
    return projects


def get_schema_specs(project_name, dataset_name=None):
    """Get the specifications of a dataset as specified in the schema.

    :param project_name: Name of project
    :param dataset_name: name of the dataset for which to get the schema. If \
            None (default), schema for all datasets is returned.
    :type project_name: str
    :type dataset_name: str
    :return: schema for dataset
    :rtype: dict
    :Example:

    >>> get_schema_specs('skynet')
    {'sarah connor': {'path': '/path/to/sarah_connor.csv',
                      'delimiter': ','},
     'kyle reese': {'path': '/path/to/kyle_reese.tsv',
                    'delimiter':, '\t'}
     'john connor': {'path': '/path/to/john_connor.txt',
                    'delimiter':, ' '}
     }
    """
    schema_file = get_default_specfile(project_name)
    with open(schema_file, "r") as f:
        specs = yaml.load(f, Loader=Loader)
    if dataset_name is not None:
        return specs[dataset_name]
    return specs


def set_schema_specs(project_name, dataset_name, **kwargs):
    """Set the schema specifications for a dataset.

    :param project_name: Name of the project containing the dataset.
    :param dataset_name: Name of the dataset of which the schema is being set.
    :param kwargs: Schema fields that are dumped into the schema files.
    :type project_name: str
    :type dataset_name: str
    :return: None
    :Example:

    >>> set_schema_specs('skynet', 'kyle reese',
                         path='/path/to/new/file.csv', delimiter=new_delimiter)
    """
    schema_file = get_default_specfile(project_name)
    with open(schema_file, "r") as f:
        specs = yaml.load(f, Loader=Loader)
    for key, value in kwargs.iteritems():
        specs[dataset_name][key] = value
    with open(schema_file, "w") as f:
        yaml.dump(specs, f, Dumper=Dumper, default_flow_style=False)


def view_projects():
    """View a list of all projects currently registered with pysemantic.

    :Example:

    >>> view_projects()
    Project skynet with specfile at /path/to/skynet.yaml
    Project south park with specfile at /path/to/south_park.yaml
    """
    projects = get_projects()
    if len(projects) > 0:
        for project_name, specfile in projects:
            print "Project {0} with specfile at {1}".format(project_name,
                                                            specfile)
    else:
        msg = textwrap.dedent("""\
            No projects found. You can add projects using the
            $ semantic list
            command.
            """)
        print msg


def remove_project(project_name):
    """Remove a project from the global configuration file.

    :param project_name: Name of the project to remove.
    :type project_name: str
    :return: True if the project existed
    :rtype: bool
    :Example:

    >>> view_projects()
    Project skynet with specfile at /path/to/skynet.yaml
    Project south park with specfile at /path/to/south_park.yaml
    >>> remove_project('skynet')
    >>> view_projects()
    Project south park with specfile at /path/to/south_park.yaml
    """
    path = locate_config_file()
    parser = RawConfigParser()
    parser.read(path)
    result = parser.remove_section(project_name)
    if result:
        with open(path, "w") as f:
            parser.write(f)
    return result


class Project(object):

    """The Project class, the entry point for most things in this module."""

    def __init__(self, project_name=None, parser=None, schema=None):
        """The Project class.

        :param project_name: Name of the project as specified in the \
                pysemantic configuration file. If this is ``None``, then the
                ``schema`` parameter is expected to contain the schema
                dictionary. (see below)
        :param parser: The parser to be used for reading dataset files. The \
                default is `pandas.read_table`.
        :param schema: Dictionary containing the schema for the project. When
        this argument is supplied (not ``None``), the ``project_name`` is
        ignored, no specfile is read, and all the specifications for the data
        are inferred from this dictionary.
        """
        if project_name is not None:
            setup_logging(project_name)
            self.project_name = project_name
            self.specfile = get_default_specfile(self.project_name)
            logger.info("Schema for project {0} found at {1}".format(project_name,
                                                                    self.specfile))
        else:
            setup_logging("no_name")
            logger.info("Schema defined by user at runtime. Not reading any "
                    "specfile.")
            self.specfile = None
        self.validators = {}
        if parser is not None:
            self.user_specified_parser = True
        else:
            self.user_specified_parser = False
        self.parser = parser
        if self.specfile is not None:
            with open(self.specfile, 'r') as f:
                specifications = yaml.load(f, Loader=Loader)
        else:
            specifications = schema
        self.column_rules = {}
        self.df_rules = {}
        for name, specs in specifications.iteritems():
            logger.info("Schema for dataset {0}:".format(name))
            logger.info(json.dumps(specs, cls=TypeEncoder))
            is_pickled = specs.get('pickle', False)
            if self.specfile is not None:
                self.validators[name] = SchemaValidator(specification=specs,
                                                        specfile=self.specfile,
                                                        name=name,
                                                        is_pickled=is_pickled)
            else:
                self.validators[name] = SchemaValidator(specification=specs,
                                                        name=name,
                                                        is_pickled=is_pickled)
            self.column_rules[name] = specs.get('column_rules', {})
            self.df_rules[name] = specs.get('dataframe_rules', {})
        self.specifications = specifications

    def export_dataset(self, dataset_name, dataframe=None, outpath=None):
        """Export a dataset to an exporter defined in the schema. If nothing is
        specified in the schema, simply export to a CSV file such named
        <dataset_name>.csv

        :param dataset_name: Name of the dataset to exporter.
        :param dataframe: Pandas dataframe to export. If None (default), this \
                dataframe is loaded using the `load_dataset` method.
        :type dataset_name: Str
        """
        if dataframe is None:
            dataframe = self.load_dataset(dataset_name)
        config = self.specifications[dataset_name].get('exporter')
        if outpath is None:
            outpath = dataset_name + ".csv"
        if config is not None:
            if config['kind'] == "aerospike":
                config['namespace'] = self.project_name
                config['set'] = dataset_name
                exporter = AerospikeExporter(config, dataframe)
                exporter.run()
        else:
            suffix = outpath.split('.')[-1]
            if suffix in ("h5", "hdf"):
                group = r'/{0}/{1}'.format(self.project_name, dataset_name)
                dataframe.to_hdf(outpath, group)
            elif suffix == "csv":
                dataframe.to_csv(outpath, index=False)

    def reload_data_dict(self):
        """Reload the data dictionary and re-populate the schema."""

        with open(self.specfile, "r") as f:
            specifications = yaml.load(f, Loader=Loader)
        self.validators = {}
        self.column_rules = {}
        self.df_rules = {}
        logger.info("Reloading project information.")
        for name, specs in specifications.iteritems():
            logger.info("Schema for dataset {0}:".format(name))
            logger.info(json.dumps(specs, cls=TypeEncoder))
            is_pickled = specs.get('pickle', False)
            self.validators[name] = SchemaValidator(specification=specs,
                                                    specfile=self.specfile,
                                                    name=name,
                                                    is_pickled=is_pickled)
            self.column_rules[name] = specs.get('column_rules', {})
            self.df_rules[name] = specs.get('dataframe_rules', {})
        self.specifications = specifications

    @property
    def datasets(self):
        """"List the datasets registered under the parent project.

        :Example:

        >>> project = Project('skynet')
        >>> project.datasets
        ['sarah connor', 'john connor', 'kyle reese']
        """
        return self.validators.keys()

    def get_dataset_specs(self, dataset_name):
        """Returns the specifications for the specified dataset in the project.

        :param dataset_name: Name of the dataset
        :type dataset_name: str
        :return: Parser arguments required to import the dataset in pandas.
        :rtype: dict
        """
        return self.validators[dataset_name].get_parser_args()

    def get_project_specs(self):
        """Returns a dictionary containing the schema for all datasets listed
        under this project.

        :return: Parser arguments for all datasets listed under the project.
        :rtype: dict
        """
        specs = {}
        for name, validator in self.validators.iteritems():
            specs[name] = validator.get_parser_args()
        return specs

    def view_dataset_specs(self, dataset_name):
        """Pretty print the specifications for a dataset.

        :param dataset_name: Name of the dataset
        :type dataset_name: str
        """
        specs = self.get_dataset_specs(dataset_name)
        pprint.pprint(specs)

    def update_dataset(self, dataset_name, dataframe, path=None, **kwargs):
        """This is tricky."""
        org_specs = self.get_dataset_specs(dataset_name)
        if path is None:
            path = org_specs['filepath_or_buffer']
        sep = kwargs.get('sep', org_specs['sep'])
        index = kwargs.get('index', False)
        dataframe.to_csv(path, sep=sep, index=index)
        dtypes = {}
        for col in dataframe:
            dtype = dataframe[col].dtype
            if dtype == np.dtype('O'):
                dtypes[col] = str
            elif dtype == np.dtype('float'):
                dtypes[col] = float
            elif dtype == np.dtype('int'):
                dtypes[col] = int
            else:
                dtypes[col] = dtype
        new_specs = {'path': path, 'delimiter': sep, 'dtypes': dtypes}
        with open(self.specfile, "r") as fid:
            specs = yaml.load(fid, Loader=Loader)
        dataset_specs = specs[dataset_name]
        dataset_specs.update(new_specs)
        if "column_rules" in dataset_specs:
            col_rules = dataset_specs['column_rules']
            cols_to_remove = []
            for colname in col_rules.iterkeys():
                if colname not in dataframe.columns:
                    cols_to_remove.append(colname)
            for colname in cols_to_remove:
                del col_rules[colname]
        logger.info("Attempting to update schema for dataset {0} to:".format(
                                                                 dataset_name))
        logger.info(json.dumps(dataset_specs, cls=TypeEncoder))
        with open(self.specfile, "w") as fid:
            yaml.dump(specs, fid, Dumper=Dumper,
                      default_flow_style=False)

    def load_dataset(self, dataset_name):
        """Load and return a dataset.

        :param dataset_name: Name of the dataset
        :type dataset_name: str
        :return: A pandas DataFrame containing the dataset.
        :rtype: pandas.DataFrame
        :Example:

        >>> demo_project = Project('pysemantic_demo')
        >>> iris = demo_project.load_dataset('iris')
        >>> type(iris)
        pandas.core.DataFrame
        """
        validator = self.validators[dataset_name]
        column_rules = self.column_rules.get(dataset_name, {})
        df_rules = self.df_rules.get(dataset_name, {})
        parser_args = validator.get_parser_args()
        df_rules.update(validator.df_rules)
        logger.info("Attempting to load dataset {} with args:".format(
                                                                 dataset_name))
        if validator.is_spreadsheet:
            parser_args.pop('usecols', None)
        logger.info(json.dumps(parser_args, cls=TypeEncoder))
        if isinstance(parser_args, dict):
            if validator.is_mysql:
                df = pd.read_sql_table(**parser_args)
            else:
                with ParseErrorHandler(parser_args, self) as handler:
                    df = handler.load()
            if df is None:
                raise ParserArgumentError("No valid parser arguments were " +
                                          "inferred from the schema.")
            if validator.is_spreadsheet and isinstance(validator.sheetname,
                                                       list):
                df = pd.concat(df.itervalues(), axis=0)
            logger.info("Success!")
            df_validator = DataFrameValidator(data=df, rules=df_rules,
                                             column_rules=column_rules)
            logger.info("Commence cleaning dataset:")
            logger.info("DataFrame rules:")
            logger.info(json.dumps(df_rules, cls=TypeEncoder))
            logger.info("Column rules:")
            logger.info(json.dumps(column_rules, cls=TypeEncoder))
            return df_validator.clean()
        else:
            dfs = []
            for argset in parser_args:
                with ParseErrorHandler(argset, self) as handler:
                    _df = handler.load()
                df_validator = DataFrameValidator(data=_df,
                                                  column_rules=column_rules)
                dfs.append(df_validator.clean())
            df = pd.concat(dfs, axis=0)
            return df.set_index(np.arange(df.shape[0]))

    def load_datasets(self):
        """Load and return all datasets.

        :return: dictionary like {dataset_name: dataframe}
        :rtype: dict
        """
        datasets = {}
        for name in self.validators.iterkeys():
            datasets[name] = self.load_dataset(name)
        return datasets
