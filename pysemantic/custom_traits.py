#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2015 jaidev <jaidev@newton>
#
# Distributed under terms of the BSD 3-clause license.

"""Customized traits for advanced validation."""

import os.path as op

from traits.api import File, List


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
        if validated_value and op.isabs(validated_value) and op.isfile(value):
            return validated_value

        self.error(obj, name, value)
