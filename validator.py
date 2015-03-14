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
from custom_traits import DTypesDict, NaturalNumber
import yaml


class DataDictValidator(HasTraits):

    # Public traits

    # Path to the data dictionary
    specfile = File(exists=True)

    # Name of the dataset described in the data dictionary
    name = Str

    # Dict trait that holds the properties of the dataset
    specification = Property(Dict, depends_on=['specfile'])

    # Path to the file containing the data
    filepath = File(exists=True)

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

    # Protected traits

    _dtypes = Property(Dict, depends_on=['specification'])

    _filepath = Property(File(exists=True), depends_on=['specification'])

    _delimiter = Property(Str, depends_on=['specification'])

    _nrows = Property(Int, depends_on=['specification'])

    _ncols = Property(Int, depends_on=['specification'])

    # Property getters and setters

    @cached_property
    def _get_specification(self):
        with open(self.specfile, "r") as f:
            data = yaml.load(f, Loader=yaml.CLoader)[self.name]
        return data

    @cached_property
    def _get_delimiter(self):
        return self.specification['delimiter']

    @cached_property
    def _get_colnames(self):
        return self.dtypes.keys()

    @cached_property
    def _get__filepath(self):
        return self.specification['path']

    @cached_property
    def _get__nrows(self):
        return self.specification['nrows']

    @cached_property
    def _get__ncols(self):
        return self.specification['ncols']

    @cached_property
    def _get__dtypes(self):
        return self.specification['dtypes']

    @cached_property
    def _get__delimiter(self):
        return self.specification['delimiter']

    # Trait change handlers

    def __dtypes_changed(self):
        """ Required because Dictionary traits that are properties don't seem
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
        """ Required because Str traits that are properties don't seem
        to do proper validation."""
        self.nrows = self._nrows

    def __ncols_changed(self):
        """ Required because Str traits that are properties don't seem
        to do proper validation."""
        self.ncols = self._ncols


if __name__ == '__main__':
    validator = DataDictValidator(specfile="dictionary.yaml", name="mtlogs")
