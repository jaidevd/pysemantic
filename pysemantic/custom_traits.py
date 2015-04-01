#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2015 jaidev <jaidev@newton>
#
# Distributed under terms of the MIT license.

"""Customized traits for advanced validation."""

import os.path as op

from traits.api import Dict, TraitError, BaseInt, File, List
from traits.trait_handlers import TraitDictObject


class ValidTraitList(List):

    """A List trait whose every element should be valid trait."""

    def validate(self, obj, name, value):
        validated_value = super(ValidTraitList, self).validate(obj, name,
                                                               value)
        for trait_name in validated_value:
            trait = obj.trait(trait_name)
            trait.validate(obj, trait_name, getattr(obj, trait_name))
        return validated_value


class AbsFile(File):

    """A File trait whose value must be an absolute path, to an existing
    file.
    """

    def validate(self, obj, name, value):
        validated_value = super(AbsFile, self).validate(obj, name, value)
        if op.isabs(validated_value) and op.isfile(value):
            return validated_value

        self.error(obj, name, value)


class NaturalNumber(BaseInt):

    """An integer trait whose value is a natural number."""

    default_value = 1

    def error(self, obj, name, value):
        msg = "The {0} trait of a {1} has to be a".format(name, obj) + \
              " value greater than zero"
        raise TraitError(args=(msg,))

    def validate(self, obj, name, value):
        value = super(NaturalNumber, self).validate(obj, name, value)
        if value > 0:
            return value
        self.error(obj, name, value)


class DTypeTraitDictObject(TraitDictObject):

    """Subclassed from the parent to aid the validation of DTypesDicts."""

    def _validate_dic(self, dic):
        """ Subclassed from parent to print a more precise TraitError"""
        name = self.name
        new_dic = {}

        key_validate = self.trait.key_trait.handler.validate
        if key_validate is None:
            key_validate = lambda obj, name, key: key

        value_validate = self.trait.value_trait.handler.validate
        if value_validate is None:
            value_validate = lambda obj, name, value: value

        obj = self.object()
        for key, value in dic.iteritems():
            try:
                key = key_validate(obj, name, key)
            except TraitError, excp:
                msg = "The name of each column in the dataset should be" + \
                      " written in the dictionary as a string."
                excp.args = (msg,)
                raise excp

            try:
                value = value_validate(obj, name, value)
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
    `pandas.read_table`.
    """

    def validate(self, obj, name, value):
        """ Subclassed from the parent to return a `DTypeTraitDictObject`
        instead of `traits.trait_handlers.TraitDictObhject`. """
        if isinstance(value, dict):
            if obj is None:
                return value
            return DTypeTraitDictObject(self, obj, name, value)
        self.error(obj, name, value)
