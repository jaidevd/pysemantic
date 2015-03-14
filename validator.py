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
                        cached_property, TraitError)
from traits.trait_handlers import TraitDictObject
import yaml


class DTypeTraitDictObject(TraitDictObject):

    def _validate_dic(self, dic):
        """ Subclassed from parent to print a more precise TraitError"""
        name = self.name
        new_dic = {}

        key_validate = self.trait.key_trait.handler.validate
        if key_validate is None:
            key_validate = lambda object, name, key: key

        value_validate = self.trait.value_trait.handler.validate
        if value_validate is None:
            value_validate = lambda object, name, value: value

        object = self.object()
        for key, value in dic.iteritems():
            try:
                key = key_validate(object, name, key)
            except TraitError, excp:
                msg = "The name of each column in the dataset should be" + \
                      " written in the dictionary as a string."
                excp.args = (msg,)
                raise excp

            try:
                value = value_validate(object, name, value)
            except TraitError, excp:
                msg = "The data type of each column in the dataset should " + \
                      "specified as a valid Python type."
                excp.args = (msg,)
                raise excp

            new_dic[key] = value

        return new_dic


class DTypesDict(Dict):

    def validate(self, object, name, value):
        """ Subclassed from the parent to return a `DTypeTraitDictObject`
        instead of `traits.trait_handlers.TraitDictObhject`. """
        if isinstance(value, dict):
            if object is None:
                return value
            return DTypeTraitDictObject(self, object, name, value)
        self.error(object, name, value)


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
