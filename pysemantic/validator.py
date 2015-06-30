#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2015 jaidev <jaidev@newton>
#
# Distributed under terms of the MIT license.

"""Traited Data validator for `pandas.DataFrame` objects."""

import copy
import cPickle
import json
import logging
import datetime
import warnings
import os.path as op

import yaml
import numpy as np
import pandas as pd
from traits.api import (HasTraits, File, Property, Str, Dict, List, Type,
                        Bool, Either, push_exception_handler, cached_property,
                        Array, Instance, Float, Any, Callable)

from pysemantic.utils import TypeEncoder, get_md5_checksum, colnames
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

    # Names of columns to be rewritten
    column_names = Property(Any, depends_on=['rules'])

    # Specifications relating to the selection of rows.
    nrows = Property(Any, depends_on=['rules'])

    def _rules_default(self):
        return {}

    @cached_property
    def _get_nrows(self):
        return self.rules.get('nrows', {})

    @cached_property
    def _get_is_drop_na(self):
        return self.rules.get("drop_na", True)

    @cached_property
    def _get_is_drop_duplicates(self):
        return self.rules.get("drop_duplicates", True)

    @cached_property
    def _get_column_names(self):
        return self.rules.get("column_names")

    def rename_columns(self):
        """Rename columns in dataframe as per the schema."""
        if self.column_names is not None:
            logger.info("Renaming columns as follows:")
            logger.info(json.dumps(self.column_names, cls=TypeEncoder))
            if isinstance(self.column_names, dict):
                for old_name, new_name in self.column_names.iteritems():
                    if old_name in self.data:
                        self.data[new_name] = self.data.pop(old_name)
            elif callable(self.column_names):
                columns = self.data.columns.copy()
                for old_name in columns:
                    new_name = self.column_names(old_name)
                    self.data[new_name] = self.data.pop(old_name)

    def clean(self):
        """Return the converted dataframe after enforcing all rules."""
        if isinstance(self.nrows, dict):
            if len(self.nrows) > 0:
                if self.nrows.get('random', False):
                    ix = self.data.index.values.copy()
                    np.random.shuffle(ix)
                    self.data = self.data.ix[ix]
                count = self.nrows.get('count', self.data.shape[0])
                self.data = self.data.ix[self.data.index[:count]]
        elif callable(self.nrows):
            ix = self.nrows(self.data.index)
            self.data = self.data.ix[self.data.index[ix]]

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
            # self.data.dropna(inplace=True)
        self.rename_columns()

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

    # Minimum value permitted in the series
    minimum = Property(Float, depends_on=['rules'])

    # Maximum value permitted in the series
    maximum = Property(Float, depends_on=['rules'])

    # Regular expression match for series containing strings
    regex = Property(Str, depends_on=['rules'])

    # List of postprocessors that work in the series
    postprocessors = Property(List, depends_on=['rules'])

    def do_postprocessing(self):
        for postprocessor in self.postprocessors:
            org_len = self.data.shape[0]
            logger.info("Applying postprocessor on column:")
            logger.info(json.dumps(postprocessor, cls=TypeEncoder))
            self.data = postprocessor(self.data)
            final_len = self.data.shape[0]
            if org_len != final_len:
                msg = ("Size of column changed after applying postprocessor."
                       "This could disturb the alignment of your data.")
                logger.warn(msg)
                warnings.warn(msg, UserWarning)

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

#    def apply_converters(self):
#        """Apply the converter functions on the series."""
#        if len(self.converters) > 0:
#            for converter in self.converters:
#                logger.info("Applying converter {0}".format(converter))
#                self.data = converter(self.data)

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
        self.do_postprocessing()
        self.apply_minmax_rules()
        self.apply_regex()
        return self.data

    @cached_property
    def _get_postprocessors(self):
        return self.rules.get("postprocessors", [])

    @cached_property
    def _get_exclude_values(self):
        return self.rules.get("exclude", [])

    @cached_property
    def _get_unique_values(self):
        return self.rules.get("unique_values", self.data.unique())

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
    def from_specfile(cls, specfile, name, **kwargs):
        """Get a validator from a schema file.

        :param specfile: Path to the schema file.
        :param name: Name of the project to create the validator for.
        """
        return cls(specfile=specfile, name=name, **kwargs)

    def __init__(self, **kwargs):
        """Overwritten to ensure that the `required_args` trait is validated
        when the object is created, not when the trait is accessed.
        """
        super(SchemaValidator, self).__init__(**kwargs)
        if not kwargs.get('is_pickled', False):
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

    # Whether the dataset is contained in a spreadsheet
    is_spreadsheet = Property(Bool, depends_on=['filepath'])

    # Delimiter
    delimiter = Str

    # number of rows in the dataset
    nrows = Either(NaturalNumber, List(NaturalNumber), Dict, Callable)

    # A dictionary whose keys are the names of the columns in the dataset, and
    # the keys are the datatypes of the corresponding columns
    dtypes = DTypesDict(key_trait=Str, value_trait=Type)

    # Names of the columns in the dataset. This is just a convenience trait,
    # it's value is just a list of the keys of `dtypes`
    colnames = Property(List, depends_on=['specification'])

    # md5 checksum of the dataset file
    md5 = Property(Str, depends_on=['filepath'])

    # List of values that represent NAs
    na_values = Property(Dict, depends_on=['specification'])

    # List of columns to combine
    datetime_cols = Property(Any, depends_on=['specification'])

    # List of converters to be applied to the columns. All converters are
    # assumed to be callables, which take the series as input and return a
    # series.
    converters = Property(Dict, depends_on=['specification'])

    # Header of the file
    header = Property(Any, depends_on=['specification'])

    # Names to use for columns in the dataframe
    column_names = Property(Any, depends_on=['specification'])

    # Rules for the dataframe that can only be enforeced after loading the
    # dataset, therefore must be exported to DataFrameValidator.
    df_rules = Dict

    # List of columns to exclude from the data
    exclude_columns = Property(List, depends_on=['specification'])

    # Whether pickled arguments exist in the schema
    is_pickled = Bool

    # Path to pickle file containing parser arguments
    pickle_file = Property(AbsFile, depends_on=['specification'])

    # Dictionary of arguments loaded from the pickle file.
    pickled_args = Property(Dict, depends_on=['pickle_file'])

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

    _nrows = Property(Any, depends_on=['specification'])

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
        if not self.is_pickled:
            if isinstance(self.filepath, list):
                if len(self.filepath) > 1:
                    return True
        return False

    @cached_property
    def _get_is_spreadsheet(self):
        if (not self.is_multifile) and (not self.is_pickled):
            return self.filepath.endswith('.xls') or self.filepath.endswith('xlsx')
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
        if not self.is_spreadsheet:
            args['error_bad_lines'] = False
        if self._delimiter:
            args['sep'] = self._delimiter

        # Columns to use
        if len(self.colnames) > 0:
            args['usecols'] = self.colnames

        # Columns to exclude
        if len(self.exclude_columns) > 0:
            usecols = colnames(self._filepath, sep=args.get('sep', ','))
            for colname in self.exclude_columns:
                usecols.remove(colname)
            args['usecols'] = usecols

        # NA values
        if len(self.na_values) > 0:
            args['na_values'] = self.na_values

        # Date/Time arguments
        # FIXME: Allow for a mix of datetime column groupings and individual
        # columns
        if len(self.datetime_cols) > 0:
            if isinstance(self.datetime_cols, dict):
                args['parse_dates'] = self.datetime_cols
            elif isinstance(self.datetime_cols, list):
                args['parse_dates'] = [self.datetime_cols]
        else:
            parse_dates = []
            for k, v in self._dtypes.iteritems():
                if v is datetime.date:
                    parse_dates.append(k)
            for k in parse_dates:
                del self._dtypes[k]
            args['dtype'] = self.dtypes
            if len(parse_dates) > 0:
                args['parse_dates'] = parse_dates

        if len(self.converters) > 0:
            args['converters'] = self.converters

        if self.header != 0:
            args['header'] = self.header
        if self.column_names is not None:
            if isinstance(self.column_names, list):
                args['names'] = self.column_names
                # Force include the header argument
                args['header'] = self.header
            elif isinstance(self.column_names, dict) or callable(self.column_names):
                self.df_rules['column_names'] = self.column_names

        if self.is_multifile:
            arglist = []
            for i in range(len(self._filepath)):
                argset = copy.deepcopy(args)
                argset.update({'filepath_or_buffer': self._filepath[i]})
                argset.update({'nrows': self._nrows[i]})
                arglist.append(argset)
            return arglist
        else:
            if self._filepath:
                args.update({'filepath_or_buffer': self._filepath})
            if "nrows" in self.specification:
                if isinstance(self._nrows, int):
                    args.update({'nrows': self._nrows})
                elif isinstance(self._nrows, dict):
                    if self._nrows.get('random', False):
                        self.df_rules.update({'nrows': self._nrows})
                    if "range" in self._nrows:
                        start, stop = self._nrows['range']
                        args['skiprows'] = start
                        args['nrows'] = stop - start
                elif callable(self._nrows):
                    self.df_rules.update({'nrows': self._nrows})
            self.pickled_args.update(args)
            if self.is_spreadsheet:
                self.pickled_args.pop('sep', None)
                self.pickled_args.pop('dtype', None)
                self.pickled_args['sheetname'] = self.name
                self.pickled_args['io'] = self.pickled_args.pop('filepath_or_buffer')
            return self.pickled_args

    def _set_parser_args(self, specs):
        self.parser_args.update(specs)

    @cached_property
    def _get_pickle_file(self):
        return self.specification.get('pickle')

    @cached_property
    def _get_pickled_args(self):
        if self.pickle_file is not None:
            with open(self.pickle_file, "r") as fid:
                args = cPickle.load(fid)
            return args
        return {}

    @cached_property
    def _get_exclude_columns(self):
        return self.specification.get("exclude_columns", [])

    @cached_property
    def _get_datetime_cols(self):
        return self.specification.get("combine_dt_columns", {})

    @cached_property
    def _get_header(self):
        return self.specification.get("header", 0)

    @cached_property
    def _get_column_names(self):
        return self.specification.get("column_names")

    @cached_property
    def _get_converters(self):
        return self.specification.get("converters", [])

    @cached_property
    def _get_md5(self):
        return self.specification.get("md5", "")

    @cached_property
    def _get_na_values(self):
        na_values = {}
        col_rules = self.specification.get("column_rules", {})
        for colname, rules in col_rules.iteritems():
            if "na_values" in rules:
                na_values[colname] = rules['na_values']
        return na_values

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

    def _df_rules_default(self):
        return {}
