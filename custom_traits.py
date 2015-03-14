#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2015 jaidev <jaidev@newton>
#
# Distributed under terms of the MIT license.

"""

"""

from traits.api import Dict, TraitError
from traits.trait_handlers import TraitDictObject


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
