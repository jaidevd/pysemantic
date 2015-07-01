#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2015 jaidev <jaidev@newton>
#
# Distributed under terms of the BSD 3-clause license.

"""Tests for the custom_traits module."""

import unittest
import os.path as op

from traits.api import HasTraits, Either, List, Str, Type, TraitError

from pysemantic.custom_traits import (AbsFile, NaturalNumber, DTypesDict,
                                      ValidTraitList)
from pysemantic.tests.test_base import TEST_DATA_DICT


class TestCustomTraits(unittest.TestCase):

    """ Testcase for the custom_traits module. This consists purely of testing
    whether validation is happening correctly on the custom_traits.
    """

    @classmethod
    def setUpClass(cls):
        class CustomTraits(HasTraits):
            def __init__(self, **kwargs):
                super(CustomTraits, self).__init__(**kwargs)
                self.required = ['filepath', 'number', 'dtype']
            filepath = AbsFile
            number = NaturalNumber
            numberlist = Either(List(NaturalNumber), NaturalNumber)
            filelist = Either(List(AbsFile), AbsFile)
            dtype = DTypesDict(key_trait=Str, value_trait=Type)
            required = ValidTraitList(Str)

        cls.custom_traits = CustomTraits

    def setUp(self):
        self.traits = self.custom_traits(filepath=op.abspath(__file__),
                                         number=2, dtype={'a': int})
        self.setter = lambda x, y: setattr(self.traits, x, y)

    def test_validtraitlist_trait(self):
        """Test if `pysemantic.self.traits.ValidTraitsList` works properly."""
        self.assertItemsEqual(self.traits.required, ['filepath', 'number',
                                                     'dtype'])

    def test_natural_number_either_list_trait(self):
        """Test of the NaturalNumber trait works within Either and List traits.
        """
        self.traits.numberlist = 1
        self.traits.numberlist = [1, 2]
        self.assertRaises(TraitError, self.setter, "numberlist", 0)
        self.assertRaises(TraitError, self.setter, "numberlist", [0, 1])

    def test_absfile_either_list_traits(self):
        """Test if the AbsFile trait works within Either and List self.traits.
        """
        self.traits.filelist = op.abspath(__file__)
        self.traits.filelist = [op.abspath(__file__), TEST_DATA_DICT]
        self.assertRaises(TraitError, self.setter, "filelist",
                          [op.basename(__file__)])
        self.assertRaises(TraitError, self.setter, "filelist", ["/foo/bar"])
        self.assertRaises(TraitError, self.setter, "filelist",
                          op.basename(__file__))
        self.assertRaises(TraitError, self.setter, "filelist", "/foo/bar")

    def test_absolute_path_file_trait(self):
        """Test if the `traits.AbsFile` trait works correctly."""
        self.traits.filepath = op.abspath(__file__)
        self.assertRaises(TraitError, self.setter, "filepath",
                          op.basename(__file__))
        self.assertRaises(TraitError, self.setter, "filepath", "foo/bar")
        self.assertRaises(TraitError, self.setter, "filepath", "/foo/bar")

    def test_natural_number_trait(self):
        """Test if the `traits.NaturalNumber` trait works correctly."""
        self.traits.number = 1
        self.assertRaises(TraitError, self.setter, "number", 0)
        self.assertRaises(TraitError, self.setter, "number", -1)

    def test_dtypes_dict_trait(self):
        """Test if the `traits.DTypesDict` trait works correctly."""
        self.traits.dtype = {'foo': int, 'bar': str, 'baz': float}
        self.assertRaises(TraitError, self.setter, "dtype", {'foo': 1})
        self.assertRaises(TraitError, self.setter, "dtype", {1: float})
        self.assertRaises(TraitError, self.setter, "dtype", {"bar": "foo"})

if __name__ == '__main__':
    unittest.main()
