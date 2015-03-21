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
                        cached_property)
from custom_traits import DTypesDict, NaturalNumber, AbsFile
import yaml
import datetime
import os.path as op


class DataDictValidator(HasTraits):

    # Public traits

    # Path to the data dictionary
    specfile = File(exists=True)

    # Name of the dataset described in the data dictionary
    name = Str

    # Dict trait that holds the properties of the dataset
    specification = Dict

    # Path to the file containing the data
    filepath = AbsFile(exists=True)

    # Delimiter
    delimiter = Str

    # number of rows in the dataset
    nrows = NaturalNumber

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

    _filepath = Property(File(exists=True), depends_on=['specification'])

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
    def _get_parser_args(self):
        args = {'filepath_or_buffer': self._filepath,
                'sep': self._delimiter,
                'nrows': self._nrows,
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
        return args

    def _set_parser_args(self, specs):
        self.parser_args.update(specs)

    @cached_property
    def _get_delimiter(self):
        return self.specification.get('delimiter', '')

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


if __name__ == '__main__':
    import pandas as pd
    specfile = "dictionary.yaml"
    with open(specfile, "r") as f:
        data = yaml.load(f, Loader=yaml.CLoader)
    datasets = {}
    for k, v in data.iteritems():
        val = DataDictValidator(specification=v, name=k)
        datasets[k] = pd.read_table(**val.get_parser_args())
