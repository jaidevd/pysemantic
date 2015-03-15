#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2015 jaidev <jaidev@newton>
#
# Distributed under terms of the MIT license.

"""
Tests
"""

import unittest
import yaml
import os.path as op
from validator import DataDictValidator


def _get_iris_args():
    filepath = op.join(op.dirname(__file__), "testdata", "iris.csv")
    return dict(filepath_or_buffer=op.abspath(filepath),
                sep=",", nrows=150,
                dtype={'Petal Length': float,
                       'Petal Width': float,
                       'Sepal Length': float,
                       'Sepal Width': float,
                       'Species': str},
                usecols=['Petal Length', 'Sepal Length', 'Petal Width',
                         'Sepal Width', 'Species'])


def _get_person_activity_args():
    filepath = op.join(op.dirname(__file__), "testdata", "person_activity.tsv")
    return dict(filepath_or_buffer=op.abspath(filepath),
                sep="\t", nrows=100, dtype={'sequence_name': str,
                                            'tag': str,
                                            'x': float,
                                            'y': float,
                                            'z': float,
                                            'activity': str},
                usecols=['sequence_name', 'tag', 'date', 'x', 'y', 'z',
                         'activity'],
                parse_dates=['date'])


class TestDataDictValidator(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.maxDiff = None
        cls.specfile = "testdata/test_dictionary.yaml"
        with open(cls.specfile, "r") as f:
            cls.basespecs = yaml.load(f, Loader=yaml.CLoader)

        # fix the paths in basespecs if they aren't absolute
        for name, dataspec in cls.basespecs.iteritems():
            if not op.isabs(dataspec['path']):
                dataspec['path'] = op.abspath(dataspec['path'])

        cls.ideal_activity_parser_args = _get_person_activity_args()
        cls.ideal_iris_parser_args = _get_iris_args()

    def assertKwargsEqual(self, dict1, dict2):
        self.assertEqual(len(dict1.keys()), len(dict2.keys()))
        for k, v in dict1.iteritems():
            self.assertIn(k, dict2)
            left = v
            right = dict2[k]
            if isinstance(left, (tuple, list)):
                self.assertItemsEqual(left, right)
            elif isinstance(left, dict):
                self.assertDictEqual(left, right)
            else:
                self.assertEqual(left, right)

    def test_validator_with_specdict_iris(self):
        """Check if the validator works when only the specification is supplied
        as a dictionary for the iris dataset."""
        validator = DataDictValidator(specification=self.basespecs['iris'])
        validated_parser_args = validator.get_parser_args()
        self.assertKwargsEqual(validated_parser_args,
                               self.ideal_iris_parser_args)

    def test_validator_with_specdist_activity(self):
        """Check if the validator works when only the specification is supplied
        as a dictionary for the person activity dataset."""
        validator = DataDictValidator(
                               specification=self.basespecs['person_activity'])
        validated_parser_args = validator.get_parser_args()
        self.assertKwargsEqual(validated_parser_args,
                               self.ideal_activity_parser_args)

    def test_error_for_relative_filepath(self):
        """Test if validator raises errors when relative paths are found in the
        dictionary."""
        specs = self.basespecs['iris']
        old_path = specs['path']
        try:
            specs['path'] = op.join("testdata", "iris.csv")
            validator = DataDictValidator(specifications=specs)
            self.assertEqual(validator.filepath, "")
        finally:
            specs['path'] = old_path


if __name__ == '__main__':
    unittest.main()
