#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2015 jaidev <jaidev@newton>
#
# Distributed under terms of the BSD 3-clause license.

"""
Tests for a the pysemantic.utils module.
"""

import unittest
import os.path as op
from pysemantic.utils import colnames, get_md5_checksum


class TestUtils(unittest.TestCase):

    def setUp(self):
        self.filepath = op.join(op.abspath(op.dirname(__file__)), "testdata",
                                "iris.csv")

    def test_colnames(self):
        """Test if the column names are read correctly from a file."""
        ideal = ['Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width',
                 'Species']
        actual = colnames(self.filepath)
        self.assertItemsEqual(actual, ideal)

    def test_colnames_infer_parser_from_extension(self):
        """Test if the colnames function can infer the correct parser from the
        file extension."""
        filepath = op.join(op.abspath(op.dirname(__file__)), "testdata",
                           "person_activity.tsv")
        ideal = "sequence_name tag date x y z activity".split()
        actual = colnames(filepath)
        self.assertItemsEqual(actual, ideal)

    def test_colnames_parser_arg(self):
        """Test if the colnames are read if the parser is specified."""
        filepath = op.join(op.abspath(op.dirname(__file__)), "testdata",
                           "person_activity.tsv")
        ideal = "sequence_name tag date x y z activity".split()
        from pandas import read_table
        actual = colnames(filepath, parser=read_table)
        self.assertItemsEqual(actual, ideal)

    def test_colnames_infer_parser_from_sep(self):
        """Test if the colnames are read if the separator is specified."""
        filepath = op.join(op.abspath(op.dirname(__file__)), "testdata",
                           "person_activity.tsv")
        ideal = "sequence_name tag date x y z activity".split()
        actual = colnames(filepath, sep='\\t')
        self.assertItemsEqual(actual, ideal)

    def test_md5(self):
        """Test the md5 checksum calculator."""
        ideal = "9b3ecf3031979169c0ecc5e03cfe20a6"
        actual = get_md5_checksum(self.filepath)
        self.assertEqual(ideal, actual)

if __name__ == '__main__':
    unittest.main()
