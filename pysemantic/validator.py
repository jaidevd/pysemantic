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
                        Bool, Either, push_exception_handler, cached_property)
from custom_traits import DTypesDict, NaturalNumber, AbsFile
import yaml
import datetime
import copy
import os.path as op

push_exception_handler(lambda *args: None, reraise_exceptions=True)


class SchemaValidator(HasTraits):

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

    # Parser args for pandas
    parser_args = Property(Dict, depends_on=['filepath', 'delimiter', 'nrows',
                                             'dtypes', 'colnames'])

    # Protected traits

    _dtypes = Property(Dict, depends_on=['specification'])

    _filepath = Property(AbsFile, depends_on=['specification'])

    _delimiter = Property(Str, depends_on=['specification'])

    _nrows = Property(Int, depends_on=['specification'])

    _ncols = Property(Int, depends_on=['specification'])

    # Public interface
    def get_parser_args(self):
        return self.parser_args

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
        args['dtype'] = self._dtypes
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
                                               Loader=yaml.CLoader).get(
                                                                 self.name, {})

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
