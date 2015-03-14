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


def _ndim_validator(value):
    if int(value) > 1:
        return int(value)


class DataFrameValidator(HasTraits):

    specfile = File(exists=True)

    name = Str

    specification = Property(Dict, depends_on=['specfile'])

    filepath = File(exists=True)

    delimiter = Str

    nrows = NaturalNumber

    ncols = NaturalNumber

    dtypes = DTypesDict(key_trait=Str, value_trait=Type)

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
    def _get__filepath(self):
        return self.specification['path']

    @cached_property
    def _get_delimiter(self):
        return self.specification['delimiter']

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

    @cached_property
    def _get_colnames(self):
        return self.dtypes.keys()

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
    validator = DataFrameValidator(specfile="dictionary.yaml", name="mtlogs")
