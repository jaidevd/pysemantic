#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2015 jaidev <jaidev@newton>
#
# Distributed under terms of the MIT license.

"""
Customized traits for advanced validation
"""

import os.path as op
from traits.api import Dict, TraitError, BaseInt, File
from traits.trait_handlers import TraitDictObject


class AbsFile(File):
    """ A File trait whose value must be an absolute path, to an existing
    file."""

    def validate(self, object, name, value):
        validated_value = super(AbsFile, self).validate(object, name, value)
        if op.isabs(validated_value) and op.isfile(value):
            return validated_value

        self.error(object, name, value)


class NaturalNumber(BaseInt):
    """ An integer trait whose value is a natural number."""

    default_value = 1

    def error(self, object, name, value):
        msg = "The {0} trait of a data dictionary has to be a".format(name) + \
              " value greater than zero"
        raise TraitError(args=(msg,))

    def validate(self, object, name, value):
        value = super(NaturalNumber, self).validate(object, name, value)
        if value > 0:
            return value
        self.error(object, name, value)


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
    """ A trait whose keys are strings, and values are Type traits. Ideally
    this is the kind of dictionary that is passed as the `dtypes` argument in
    `pandas.read_table`."""

    def validate(self, object, name, value):
        """ Subclassed from the parent to return a `DTypeTraitDictObject`
        instead of `traits.trait_handlers.TraitDictObhject`. """
        if isinstance(value, dict):
            if object is None:
                return value
            return DTypeTraitDictObject(self, object, name, value)
        self.error(object, name, value)
