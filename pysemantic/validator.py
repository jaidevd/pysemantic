#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2015 jaidev <jaidev@newton>
#
# Distributed under terms of the MIT license.

"""Traited Data validator for `pandas.DataFrame` objects."""

import copy
import json
import logging
import datetime
import warnings
import os.path as op

import yaml
import numpy as np
import pandas as pd
from traits.api import (HasTraits, File, Property, Int, Str, Dict, List, Type,
                        Bool, Either, push_exception_handler, cached_property,
                        Array, Instance, Callable, Float)

from pysemantic.utils import TypeEncoder, get_md5_checksum
from pysemantic.custom_traits import (DTypesDict, NaturalNumber, AbsFile,
                                      ValidTraitList)

try:
    from yaml import CDumper as Dumper
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Dumper, Loader

push_exception_handler(lambda *args: None, reraise_exceptions=True)
logger = logging.getLogger(__name__)


class DataFrameValidator(HasTraits):

    """A validator class for `pandas.DataFrame` objects."""

    # The dataframe in question
    data = Instance(pd.DataFrame)

    # the column rules to be enforced
    column_rules = Dict

    # rules related to the dataset itself
    rules = Dict

    # whether to drop duplicates
    is_drop_duplicates = Property(Bool, depends_on=['rules'])

    # whether to drop NAs
    is_drop_na = Property(Bool, depends_on=['rules'])

    def _rules_default(self):
        return {}

    @cached_property
    def _get_is_drop_na(self):
        return self.rules.get("drop_na", True)

    @cached_property
    def _get_is_drop_duplicates(self):
        return self.rules.get("drop_duplicates", True)

    def clean(self):
        """Return the converted dataframe after enforcing all rules."""
        if self.is_drop_na:
            x = self.data.shape[0]
            self.data.dropna(inplace=True)
            y = self.data.shape[0]
            logger.info("{0} rows containing NAs were dropped.".format(x - y))

        if self.is_drop_duplicates:
            x = self.data.shape[0]
            self.data.drop_duplicates(inplace=True)
            y = self.data.shape[0]
            logger.info("{0} duplicate rows were dropped.".format(x - y))

        for col in self.data:
            logger.info("Commence cleaning of column {}".format(col))
            series = self.data[col]
            rules = self.column_rules.get(col, {})
            validator = SeriesValidator(data=series, rules=rules)
            self.data[col] = validator.clean()
            if len(validator.exclude_values) > 0:
                for exval in validator.exclude_values:
                    self.data.drop(self.data.index[self.data[col] == exval],
                                   inplace=True)
                logger.info("Excluding following values from col {0}".format(
                                                                          col))
                logger.info(json.dumps(validator.exclude_values))
            self.data.dropna(inplace=True)

        return self.data


class SeriesValidator(HasTraits):

    """A validator class for `pandas.Series` objects."""

    # the series in question
    data = Instance(pd.Series)

    # Rules of validation
    rules = Dict

    # Whether to drop NAs from the series.
    is_drop_na = Property(Bool, depends_on=['rules'])

    # Whether to drop duplicates from the series.
    is_drop_duplicates = Property(Bool, depends_on=['rules'])

    # Unique values encountered in the series
    unique_values = Property(Array, depends_on=['rules'])

    # List of values to exclude
    exclude_values = Property(List, depends_on=['rules'])

    # List of converters to be applied to the series. All converters are
    # assumed to be callables, which take the series as input and return a
    # series.
    converters = Property(List(Callable), depends_on=['rules'])

    # Minimum value permitted in the series
    minimum = Property(Float, depends_on=['rules'])

    # Maximum value permitted in the series
    maximum = Property(Float, depends_on=['rules'])

    # Regular expression match for series containing strings
    regex = Property(Str, depends_on=['rules'])

    def do_drop_duplicates(self):
        """Drop duplicates from the series if required."""
        if self.is_drop_duplicates:
            duplicates = self.data.index[self.data.duplicated()].tolist()
            logger.info("Following duplicated rows were dropped:")
            logger.info(json.dumps(duplicates))
            self.data.drop_duplicates(inplace=True)

    def do_drop_na(self):
        """Drop NAs from the series if required."""
        if self.is_drop_na:
            na_bool = pd.isnull(self.data)
            na_rows = self.data.index[na_bool].tolist()
            logger.info("Following rows containing NAs were dropped:")
            logger.info(json.dumps(na_rows))
            self.data.dropna(inplace=True)

    def apply_converters(self):
        """Apply the converter functions on the series."""
        if len(self.converters) > 0:
            for converter in self.converters:
                logger.info("Applying converter {0}".format(converter))
                self.data = converter(self.data)

    def apply_uniques(self):
        """Remove all values not included in the `uniques`."""
        if not np.all(self.data.unique() == self.unique_values):
            logger.info("Keeping only the following unique values:")
            logger.info(json.dumps(self.unique_values))
            for value in self.data.unique():
                if value not in self.unique_values:
                    self.data = self.data[self.data != value]

    def drop_excluded(self):
        """Remove all values specified in `exclude_values`."""
        if len(self.exclude_values) > 0:
            logger.info("Removing the following excluded values:")
            logger.info(json.dumps(self.excluded_values))
            for value in self.exclude_values:
                self.data.drop(self.data.index[self.data == value],
                               inplace=True)

    def apply_minmax_rules(self):
        """Restrict the series to the minimum and maximum from the schema."""
        if self.data.dtype in (int, float, datetime.date):
            if self.minimum != -np.inf:
                logger.info("Setting minimum at {0}".format(self.minimum))
                self.data = self.data[self.data >= self.minimum]
            if self.maximum != np.inf:
                logger.info("Setting maximum at {0}".format(self.maximum))
                self.data = self.data[self.data <= self.maximum]

    def apply_regex(self):
        """Apply a regex filter on strings in the series."""
        if self.regex:
            if self.data.dtype is np.dtype('O'):
                # filter by regex
                logger.info("Applying regex filter with the following regex:")
                logger.info(self.regex)
                self.data = self.data[self.data.str.contains(self.regex)]

    def clean(self):
        """Return the converted dataframe after enforcing all rules."""
        self.do_drop_duplicates()
        self.do_drop_na()
        self.apply_uniques()
        self.apply_converters()
        self.apply_minmax_rules()
        self.apply_regex()
        return self.data

    @cached_property
    def _get_exclude_values(self):
        return self.rules.get("exclude", [])

    @cached_property
    def _get_unique_values(self):
        return self.rules.get("unique_values", self.data.unique())

    @cached_property
    def _get_converters(self):
        return self.rules.get("converters", [])

    @cached_property
    def _get_is_drop_na(self):
        return self.rules.get("drop_na", False)

    @cached_property
    def _get_is_drop_duplicates(self):
        return self.rules.get("drop_duplicates", False)

    @cached_property
    def _get_minimum(self):
        return self.rules.get("min", -np.inf)

    @cached_property
    def _get_maximum(self):
        return self.rules.get("max", np.inf)

    @cached_property
    def _get_regex(self):
        return self.rules.get("regex", "")


class SchemaValidator(HasTraits):

    """A validator class for schema in the data dictionary."""

    @classmethod
    def from_dict(cls, specification):
        """Get a validator from a schema dictionary.

        :param specification: Dictionary containing schema specifications.
        """
        return cls(specification=specification)

    @classmethod
    def from_specfile(cls, specfile, name):
        """Get a validator from a schema file.

        :param specfile: Path to the schema file.
        :param name: Name of the project to create the validator for.
        """
        return cls(specfile=specfile, name=name)

    def __init__(self, **kwargs):
        """Overwritten to ensure that the `required_args` trait is validated
        when the object is created, not when the trait is accessed.
        """
        super(SchemaValidator, self).__init__(**kwargs)
        self.required_args = ['filepath', 'delimiter']

    # Public traits

    # Path to the data dictionary
    specfile = File(exists=True)

    # Name of the dataset described in the data dictionary
    name = Str

    # Dict trait that holds the properties of the dataset
    specification = Dict

    # Path to the file containing the data
    filepath = Either(AbsFile, List(AbsFile))

    # Whether the dataset spans multiple files
    is_multifile = Property(Bool, depends_on=['filepath'])

    # Delimiter
    delimiter = Str

    # number of rows in the dataset
    nrows = Either(NaturalNumber, List(NaturalNumber))

    # A dictionary whose keys are the names of the columns in the dataset, and
    # the keys are the datatypes of the corresponding columns
    dtypes = DTypesDict(key_trait=Str, value_trait=Type)

    # Names of the columns in the dataset. This is just a convenience trait,
    # it's value is just a list of the keys of `dtypes`
    colnames = Property(List, depends_on=['specification'])

    # md5 checksum of the dataset file
    md5 = Property(Str, depends_on=['filepath'])

    # List of required traits
    # FIXME: Arguments required by the schema should't have to be programmed
    # into the validator class. There must be a way to enforce requirements
    # right in the schema itself.
    required_args = ValidTraitList

    # Parser args for pandas
    parser_args = Property(Dict, depends_on=['filepath', 'delimiter', 'nrows',
                                             'dtypes', 'colnames'])

    # Protected traits

    _dtypes = Property(DTypesDict(key_trait=Str, value_trait=Type),
                       depends_on=['specification'])

    _filepath = Property(AbsFile, depends_on=['specification'])

    _delimiter = Property(Str, depends_on=['specification'])

    _nrows = Property(Int, depends_on=['specification'])

    # Public interface

    def get_parser_args(self):
        """Return parser args as required by pandas parsers."""
        return self.parser_args

    to_dict = get_parser_args

    def set_parser_args(self, specs, write_to_file=False):
        """Magic method required by Property traits."""
        self.parser_args = specs
        if write_to_file:
            logger.info("Following specs for dataset {0}".format(self.name) +
                        " were written to specfile {0}".format(self.specfile))
            with open(self.specfile, "r") as f:
                allspecs = yaml.load(f, Loader=Loader)
            allspecs[self.name] = specs
            with open(self.specfile, "w") as f:
                yaml.dump(allspecs, f, Dumper=Dumper,
                          default_flow_style=False)
        else:
            logger.info("Following parser args were set for dataset {}".format(
                                                                    self.name))
        logger.info(json.dumps(specs, cls=TypeEncoder))
        return True

    # Property getters and setters

    @cached_property
    def _get_is_multifile(self):
        if isinstance(self.filepath, list):
            if len(self.filepath) > 1:
                return True
        return False

    @cached_property
    def _get_parser_args(self):
        if self.md5:
            if self.md5 != get_md5_checksum(self.filepath):
                msg = \
                    """The MD5 checksum of the file {} does not match the one
                     specified in the schema. This may not be the file you are
                     looking for."""
                logger.warn(msg.format(self.filepath))
                warnings.warn(msg.format(self.filepath), UserWarning)
        args = {}
        if self._delimiter:
            args['sep'] = self._delimiter
        if len(self.colnames) > 0:
            args['usecols'] = self.colnames
        parse_dates = []
        for k, v in self._dtypes.iteritems():
            if v is datetime.date:
                parse_dates.append(k)
        for k in parse_dates:
            del self._dtypes[k]
        args['dtype'] = self.dtypes
        if len(parse_dates) > 0:
            args['parse_dates'] = parse_dates
        if self.is_multifile:
            arglist = []
            for i in range(len(self._filepath)):
                argset = copy.deepcopy(args)
                argset.update({'filepath_or_buffer': self._filepath[i]})
                argset.update({'nrows': self._nrows[i]})
                arglist.append(argset)
            return arglist
        else:
            args.update({'filepath_or_buffer': self._filepath})
            if "nrows" in self.specification:
                args.update({'nrows': self._nrows})
            return args

    def _set_parser_args(self, specs):
        self.parser_args.update(specs)

    @cached_property
    def _get_md5(self):
        return self.specification.get("md5", "")

    @cached_property
    def _get_colnames(self):
        return self.specification.get('use_columns', [])

    @cached_property
    def _get__filepath(self):
        return self.specification.get('path', "")

    @cached_property
    def _get__nrows(self):
        return self.specification.get('nrows', 1)

    @cached_property
    def _get__dtypes(self):
        return self.specification.get('dtypes', {})

    @cached_property
    def _get__delimiter(self):
        return self.specification.get('delimiter', '')

    # Trait change handlers

    def _specfile_changed(self):
        if self.specification == {}:
            with open(self.specfile, "r") as f:
                self.specification = yaml.load(f,
                                               Loader=Loader).get(
                                                                 self.name, {})

    def _filepath_default(self):
        return self.specification.get("path")

    def __dtypes_items_changed(self):
        self.dtypes = self._dtypes

    def __filepath_changed(self):
        self.filepath = self._filepath

    def __delimiter_changed(self):
        self.delimiter = self._delimiter

    def __nrows_changed(self):
        self.nrows = self._nrows

    # Trait initializers

    def _specification_default(self):
        if op.isfile(self.specfile):
            with open(self.specfile, 'r') as f:
                data = yaml.load(f, Loader=Loader).get(self.name, {})
            return data
        return {}

    def _dtypes_default(self):
        return self._dtypes
