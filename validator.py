#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2015 jaidev <jaidev@newton>
#
# Distributed under terms of the MIT license.

"""

"""

from traits.api import (HasTraits, File, Property, Int, Str, Dict, List, Type,
                        cached_property)
from custom_traits import DTypesDict
import yaml


class DataFrameValidator(HasTraits):

    specfile = File

    name = Str

    specification = Property(Dict, depends_on=['specfile'])

    filepath = Property(File, depends_on=['specification'])

    delimiter = Property(Str, depends_on=['specification'])

    nrows = Property(Int, depends_on=['specification'])

    ncols = Property(Int, depends_on=['specification'])

    dtypes = DTypesDict(key_trait=Str, value_trait=Type)

    colnames = Property(List, depends_on=['dtypes'])

    # Protected traits

    _dtypes = Property(Dict, depends_on=['specification'])

    # Property getters and setters

    @cached_property
    def _get_specification(self):
        with open(self.specfile, "r") as f:
            data = yaml.load(f, Loader=yaml.CLoader)[self.name]
        return data

    @cached_property
    def _get_filepath(self):
        return self.specification['path']

    @cached_property
    def _get_delimiter(self):
        return self.specification['delimiter']

    @cached_property
    def _get_nrows(self):
        return self.specification['nrows']

    @cached_property
    def _get_ncols(self):
        return self.specification['ncols']

    @cached_property
    def _get__dtypes(self):
        return self.specification['dtypes']

    @cached_property
    def _get_colnames(self):
        return self.dtypes.keys()

    def __dtypes_changed(self):
        """ Required because Dictionary traits that are properties don't seem
        to do proper validation."""
        self.dtypes = self._dtypes

if __name__ == '__main__':
    validator = DataFrameValidator(specfile="dictionary.yaml", name="mtlogs")
