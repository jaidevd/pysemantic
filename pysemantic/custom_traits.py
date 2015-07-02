#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2015 jaidev <jaidev@newton>
#
# Distributed under terms of the BSD 3-clause license.

"""Customized traits for advanced validation."""

import os.path as op

from traits.api import TraitError, BaseInt, File, List


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
