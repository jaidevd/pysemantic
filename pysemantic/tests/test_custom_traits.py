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

from traits.api import HasTraits, Either, List, Str, TraitError

from pysemantic.custom_traits import AbsFile, ValidTraitList
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
                self.required = ['filepath']
            filepath = AbsFile
            filelist = Either(List(AbsFile), AbsFile)
            required = ValidTraitList(Str)

        cls.custom_traits = CustomTraits

    def setUp(self):
        self.traits = self.custom_traits(filepath=op.abspath(__file__))
        self.setter = lambda x, y: setattr(self.traits, x, y)

    def test_validtraitlist_trait(self):
        """Test if `pysemantic.self.traits.ValidTraitsList` works properly."""
        self.assertItemsEqual(self.traits.required, ['filepath'])

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


if __name__ == '__main__':
    unittest.main()
