#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2015 jaidev <jaidev@newton>
#
# Distributed under terms of the MIT license.

"""The Project class."""

import os
import warnings
import textwrap
import pprint
from ConfigParser import RawConfigParser
import os.path as op

import yaml
import pandas as pd
import numpy as np

from pysemantic.validator import SchemaValidator, DataFrameValidator
from pysemantic.errors import MissingProject, MissingConfigError

CONF_FILE_NAME = os.environ.get("PYSEMANTIC_CONFIG", "pysemantic.conf")


def _locate_config_file():
    """Locates the configuration file used by semantic."""
    paths = [op.join(os.getcwd(), CONF_FILE_NAME),
             op.join(op.expanduser('~'), CONF_FILE_NAME)]
    for path in paths:
        if op.exists(path):
            return path
    raise MissingConfigError("No pysemantic configuration file was fount at"
                             " {0} or {1}".format(*paths))


def _get_default_specfile(project_name):
    """Returns the specifications file used by the given project. The
    configuration file is searched for first in the current directory and then
    in the home directory.

    :param project_name: Name of the project for which to get the spcfile.
    """
    path = _locate_config_file()
    parser = RawConfigParser()
    parser.read(path)
    return parser.get(project_name, 'specfile')


def add_project(project_name, specfile):
    """Add a project to the global configuration file.

    :param project_name: Name of the project
    :param specfile: path to the data dictionary used by the project.
    """
    path = _locate_config_file()
    parser = RawConfigParser()
    parser.read(path)
    parser.add_section(project_name)
    parser.set(project_name, "specfile", specfile)
    with open(path, "w") as f:
        parser.write(f)


def add_dataset(project_name, dataset_name, dataset_specs):
    """Add a dataset to a project.

    :param project_name: Name of the project to which the dataset is to be
    added.
    :param dataset_name: Name of the dataset to be added.
    :param dataset_specs: Specifications of the dataset.
    """
    data_dict = _get_default_specfile(project_name)
    with open(data_dict, "r") as f:
        spec = yaml.load(f, Loader=yaml.CLoader)
    spec[dataset_name] = dataset_specs
    with open(data_dict, "w") as f:
        yaml.dump(spec, f, Dumper=yaml.CDumper, default_flow_style=False)


def remove_dataset(project_name, dataset_name):
    """Removes a dataset from a project.

    :param project_name: Name of the project
    :param dataset_name: Name of the dataset to remove
    """
    data_dict = _get_default_specfile(project_name)
    with open(data_dict, "r") as f:
        spec = yaml.load(f, Loader=yaml.CLoader)
    del spec[dataset_name]
    with open(data_dict, "w") as f:
        yaml.dump(spec, f, Dumper=yaml.CDumper, default_flow_style=False)


def set_schema_fpath(project_name, schema_fpath):
    """Set the schema path for a given project.

    :param project_name: Name of the project
    :param schema_fpath: path to the yaml file to be used as the schema for the
    project.
    """
    path = _locate_config_file()
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
    """
    path = _locate_config_file()
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
    :param dataset_name: name of the dataset for which to get the schema. If
    None(default), schema for all datasets is returned.
    """
    schema_file = _get_default_specfile(project_name)
    with open(schema_file, "r") as f:
        specs = yaml.load(f, Loader=yaml.CLoader)
    if dataset_name is not None:
        return specs[dataset_name]
    return specs


def set_schema_specs(project_name, dataset_name, **kwargs):
    """Set the schema specifications for a dataset.

    :param project_name: Name of the project containing the dataset.
    :param dataset_name: Name of the dataset of which the schema is being set.
    :param **kwargs: Schema fields that are dumped into the schema files.
    """
    schema_file = _get_default_specfile(project_name)
    with open(schema_file, "r") as f:
        specs = yaml.load(f, Loader=yaml.CLoader)
    for key, value in kwargs.iteritems():
        specs[dataset_name][key] = value
    with open(schema_file, "w") as f:
        yaml.dump(specs, f, Dumper=yaml.CDumper, default_flow_style=False)


def view_projects():
    """View a list of all projects currently registered with pysemantic."""
    projects = get_projects()
    for project_name, specfile in projects:
        print "Project {0} with specfile at {1}".format(project_name, specfile)


def remove_project(project_name):
    """Remove a project from the global configuration file.

    :param project_name: Name of the project to remove.
    Returns true if the project existed.
    """
    path = _locate_config_file()
    parser = RawConfigParser()
    parser.read(path)
    result = parser.remove_section(project_name)
    if result:
        with open(path, "w") as f:
            parser.write(f)
    return result


class Project(object):

    """The Project class, which is the entry point for most things in this
    module.
    """

    def __init__(self, project_name, parser=None):
        """__init__

        :param project_name: Name of the project as specified in the pysemantic
        configuration file.
        :param parser: The parser to be used for reading dataset files. The
        default is `pandas.read_table`.
        """
        self.project_name = project_name
        self.specfile = _get_default_specfile(self.project_name)
        self.validators = {}
        if parser is not None:
            self.user_specified_parser = True
        else:
            self.user_specified_parser = False
        self.parser = parser
        with open(self.specfile, 'r') as f:
            specifications = yaml.load(f, Loader=yaml.CLoader)
        self.column_rules = {}
        self.df_rules = {}
        for name, specs in specifications.iteritems():
            self.validators[name] = SchemaValidator(specification=specs,
                                                    specfile=self.specfile,
                                                    name=name)
            self.column_rules[name] = specs.get('column_rules', {})
            self.df_rules[name] = specs.get('dataframe_rules', {})

    @property
    def datasets(self):
        return self.validators.keys()

    def get_dataset_specs(self, dataset_name):
        """Returns the specifications for the specified dataset in the project.

        :param dataset_name: Name of the dataset
        """
        return self.validators[dataset_name].get_parser_args()

    def get_project_specs(self):
        """Returns a dictionary containing the schema for all datasets listed
        under this project.
        """
        specs = {}
        for name, validator in self.validators.iteritems():
            specs[name] = validator.get_parser_args()
        return specs

    def view_dataset_specs(self, dataset_name):
        """Pretty print the specifications for a dataset.

        :param dataset_name: Name of the dataset
        """
        specs = self.get_dataset_specs(dataset_name)
        pprint.pprint(specs)

    def set_dataset_specs(self, dataset_name, specs, write_to_file=False):
        """Sets the specifications to the dataset. Using this is not
        recommended. All specifications for datasets should be handled through
        the data dictionary.

        :param dataset_name: Name of the dataset for which specifications need
        to be modified.
        :param specs: A dictionary containing the new specifications for the
        dataset.
        :param write_to_file: If true, the data dictionary will be updated to
        the new specifications. If False (the default), the new specifications
        are used for the respective dataset only for the lifetime of the
        `Project` object.
        """
        validator = self.validators[dataset_name]
        return validator.set_parser_args(specs, write_to_file)

    def load_dataset(self, dataset_name):
        """Load and return the dataset.

        :param dataset_name: Name of the dataset
        """
        validator = self.validators[dataset_name]
        column_rules = self.column_rules.get(dataset_name, {})
        df_rules = self.df_rules.get(dataset_name, {})
        args = validator.get_parser_args()
        if isinstance(args, dict):
            df = self._load(args)
            df_validator = DataFrameValidator(data=df, rules=df_rules,
                                             column_rules=column_rules)
            return df_validator.clean()
        else:
            dfs = []
            for argset in args:
                self._update_parser(argset)
                _df = self.parser(**argset)
                df_validator = DataFrameValidator(data=_df,
                                                  column_rules=column_rules)
                dfs.append(df_validator.clean())
            return pd.concat(dfs, axis=0)

    def load_datasets(self):
        """Loads and returns all datasets listed in the data dictionary for the
        project.
        """
        datasets = {}
        for name in self.validators.iterkeys():
            datasets[name] = self.load_dataset(name)
        return datasets

    def _update_parser(self, argdict):
        """Update the pandas parser based on the delimiter.

        :param argdict: Dictionary containing parser arguments.
        """
        if not self.user_specified_parser:
            sep = argdict['sep']
            if sep == ",":
                self.parser = pd.read_csv
            else:
                self.parser = pd.read_table

    def _load(self, parser_args):
        """The actual loader function that does the heavy lifting.

        :param parser_args: Dictionary containing parser arguments.
        """
        self._update_parser(parser_args)
        try:
            return self.parser(**parser_args)
        except ValueError as e:
            if e.message.startswith("Falling back to the 'python' engine"):
                del parser_args['dtype']
                msg = textwrap.dedent("""\
                        Dtypes are not supported regex delimiters. Ignoring the
                        dtypes in the schema. Consider fixing this by editing
                        the schema for better performance.
                        """)
                warnings.warn(msg, UserWarning)
                return self.parser(**parser_args)
            elif e.message.startswith("cannot safely convert"):
                bad_col = int(e.message.split(' ')[-1])
                bad_col = parser_args['dtype'].keys()[bad_col]
                specified_dtype = parser_args['dtype'][bad_col]
                del parser_args['dtype'][bad_col]
                msg = textwrap.dedent("""\
                The specified dtype for the column '{0}' ({1}) seems to be
                incorrect. This has been ignored for now.
                Consider fixing this by editing the schema.""".format(bad_col,
                                                              specified_dtype))
                warnings.warn(msg, UserWarning)
                return self.parser(**parser_args)
        except AttributeError as e:
            if e.message == "'NoneType' object has no attribute 'dtype'":
                bad_rows = self._detect_mismatched_dtype_row(int, parser_args)
                for col in bad_rows:
                    del parser_args['dtype'][col]
                return self.parser(**parser_args)
        except Exception as e:
            if e.message == "Integer column has NA values":
                bad_rows = self._detect_row_with_na(parser_args)
                if len(bad_rows) > 0:
                    newtypelist = [(colname, float) for colname in bad_rows]
                    self._update_dtypes(parser_args['dtype'], newtypelist)
            return self.parser(**parser_args)

    def _update_dtypes(self, dtypes, typelist):
        """Update the dtypes parameter of the parser arguments.

        :param dtypes: The original column types
        :param typelist: List of tuples [(column_name, new_dtype), ...]
        """
        for colname, coltype in typelist:
            dtypes[colname] = coltype

    def _detect_row_with_na(self, parser_args):
        """Return the list of columns in the dataframe, for which the data type
        has been marked as integer, but which contain NAs.

        :param parser_args: Dictionary containing parser arguments.
        """
        dtypes = parser_args.get("dtype")
        usecols = parser_args.get("usecols")
        int_cols = [col for col in usecols if dtypes.get(col) is int]
        fpath = parser_args['filepath_or_buffer']
        sep = parser_args['sep']
        nrows = parser_args.get('nrows')
        df = self.parser(fpath, sep=sep, usecols=int_cols, nrows=nrows)
        bad_rows = []
        for col in df:
            if np.any(pd.isnull(df[col])):
                bad_rows.append(col)
        return bad_rows

    def _detect_mismatched_dtype_row(self, specified_dtype, parser_args):
        """Check the dataframe for rows that have a badly specified dtype.

        :param specfified_dtype: The datatype specified in the schema
        :param parser_args: Dictionary containing parser arguments.
        """
        to_read = []
        dtypes = parser_args.get("dtype")
        for key, value in dtypes.iteritems():
            if value is specified_dtype:
                to_read.append(key)
        fpath = parser_args['filepath_or_buffer']
        sep = parser_args['sep']
        nrows = parser_args.get('nrows')
        df = self.parser(fpath, sep=sep, usecols=to_read, nrows=nrows)
        bad_cols = []
        for col in df:
            try:
                df[col] = df[col].astype(specified_dtype)
            except ValueError:
                bad_cols.append(col)
                msg = textwrap.dedent("""\
                The specified dtype for the column '{0}' ({1}) seems to be
                incorrect. This has been ignored for now.
                Consider fixing this by editing the schema.""".format(col,
                                                              specified_dtype))
                warnings.warn(msg, UserWarning)
        return bad_cols
