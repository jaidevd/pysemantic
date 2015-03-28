#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2015 jaidev <jaidev@newton>
#
# Distributed under terms of the MIT license.

"""
Traited Data validator for `pandas.DataFrame` objects
"""

from traits.api import (HasTraits, File, Property, Int, Str, Dict, List, Type,
                        Bool, Either, push_exception_handler, cached_property,
                        Array, Instance, Callable, Float)
from custom_traits import DTypesDict, NaturalNumber, AbsFile, ValidTraitList
import pandas as pd
import numpy as np
import yaml
import datetime
import re
import copy
import os.path as op

push_exception_handler(lambda *args: None, reraise_exceptions=True)


class DataFrameValidator(HasTraits):

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
        for col in self.data:
            series = self.data[col]
            rules = self.column_rules[col]
            validator = SeriesValidator(data=series, rules=rules)
            self.data[col] = validator.clean()
        if self.is_drop_na:
            self.data.dropna(inplace=True)
        if self.is_drop_duplicates:
            self.data.drop_duplicates(inplace=True)
        return self.data


class SeriesValidator(HasTraits):

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

    def clean(self):
        if self.is_drop_duplicates:
            self.data.drop_duplicates(inplace=True)

        if self.is_drop_na:
            self.data.dropna(inplace=True)

        if not np.all(self.data.unique() == self.unique_values):
            for value in self.data.unique():
                if value not in self.unique_values:
                    self.data = self.data[self.data != value]

        if len(self.converters) > 0:
            for converter in self.converters:
                self.data = converter(self.data)

        if self.data.dtype in (int, float, datetime.date):
            if self.minimum != -np.inf:
                self.data = self.data[self.data >= self.minimum]
            if self.maximum != np.inf:
                self.data = self.data[self.data <= self.maximum]

        if self.regex:
            if self.data.dtype is np.dtype('O'):
                # filter by regex
                re_filter = lambda x: re.search(self.regex, x)
                re_matches = self.data.apply(re_filter)
                self.data = self.data[pd.notnull(re_matches)]

        return self.data

    @cached_property
    def _get_unique_values(self):
        return self.rules.get("unique_values", self.data.unique())

    @cached_property
    def _get_converters(self):
        return self.rules.get("converters", [])

    @cached_property
    def _get_is_drop_na(self):
        return self.rules.get("drop_na", True)

    @cached_property
    def _get_is_drop_duplicates(self):
        return self.rules.get("drop_duplicates", True)

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

    @classmethod
    def from_dict(cls, specification):
        return cls(specification=specification)

    @classmethod
    def from_specfile(cls, specfile, name):
        return cls(specfile=specfile, name=name)

    def __init__(self, **kwargs):
        """Overwritten to ensure that the `required_args` trait is validated
        when the object is created, not when the trait is accessed."""
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

    # number of columns in the dataset
    ncols = NaturalNumber

    # A dictionary whose keys are the names of the columns in the dataset, and
    # the keys are the datatypes of the corresponding columns
    dtypes = DTypesDict(key_trait=Str, value_trait=Type)

    # Names of the columns in the dataset. This is just a convenience trait,
    # it's value is just a list of the keys of `dtypes`
    colnames = Property(List, depends_on=['dtypes'])

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

    _ncols = Property(Int, depends_on=['specification'])

    # Public interface

    def get_parser_args(self):
        return self.parser_args

    to_dict = get_parser_args

    def set_parser_args(self, specs, write_to_file=False):
        self.parser_args = specs
        if write_to_file:
            try:
                with open(self.specfile, "r") as f:
                    allSpecs = yaml.load(f, Loader=yaml.CLoader)
                allSpecs[self.name] = specs
                with open(self.specfile, "w") as f:
                    yaml.dump(allSpecs, f, Dumper=yaml.CDumper)
                return True
            except Exception as e:
                import warnings
                msg = ("Writing specification to file failed with the "
                       "following error - {0}.".format(e))
                warnings.warn(msg, RuntimeWarning, stacklevel=3)
                return False
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
        args = {'sep': self._delimiter,
                'usecols': self.colnames}
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

    def _get_colnames(self):
        return self._dtypes.keys()

    @cached_property
    def _get__filepath(self):
        return self.specification.get('path', "")

    @cached_property
    def _get__nrows(self):
        return self.specification.get('nrows', 1)

    @cached_property
    def _get__ncols(self):
        return self.specification.get('ncols', 0)

#    @cached_property
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
                                               Loader=yaml.CLoader).get(
                                                                 self.name, {})

    def _filepath_default(self):
        return self.specification.get("path")

    def __dtypes_items_changed(self):
        """ Required because Dict traits that are properties don't seem
        to do proper validation."""
        self.dtypes = self._dtypes

    def __filepath_changed(self):
        """ Required because File traits that are properties don't seem
        to do proper validation."""
        self.filepath = self._filepath

    def __delimiter_changed(self):
        """ Required because Str traits that are properties don't seem
        to do proper validation."""
        self.delimiter = self._delimiter

    def __nrows_changed(self):
        """ Required because Int traits that are properties don't seem
        to do proper validation."""
        self.nrows = self._nrows

    def __ncols_changed(self):
        """ Required because Int traits that are properties don't seem
        to do proper validation."""
        self.ncols = self._ncols

    # Trait initializers

    def _specification_default(self):
        if op.isfile(self.specfile):
            with open(self.specfile, 'r') as f:
                data = yaml.load(f, Loader=yaml.CLoader).get(self.name, {})
            return data
        return {}

    def _dtypes_default(self):
        return self._dtypes
