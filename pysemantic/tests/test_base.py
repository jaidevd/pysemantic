#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2015 jaidev <jaidev@newton>
#
# Distributed under terms of the MIT license.

"""Base classes and functions for tests."""

import os
import unittest
import os.path as op
from copy import deepcopy
from ConfigParser import RawConfigParser

import yaml
import numpy as np
import pandas as pd

from pysemantic import project as pr

TEST_CONFIG_FILE_PATH = op.join(op.abspath(op.dirname(__file__)), "testdata",
                                "test.conf")
TEST_DATA_DICT = op.join(op.abspath(op.dirname(__file__)), "testdata",
                         "test_dictionary.yaml")


def _path_fixer(filepath, root=None):
    """Change all the relative paths in `filepath` to absolute ones.

    :param filepath: File to be changed
    :param root: Root path with which the relative paths are prefixed. If None
    (default), the directory with this file is the root.
    """
    if root is None:
        root = op.join(op.abspath(op.dirname(__file__)))
    if filepath.endswith((".yaml", ".yml")):
        with open(filepath, "r") as fileobj:
            data = yaml.load(fileobj, Loader=yaml.CLoader)
        for specs in data.itervalues():
            specs['path'] = op.join(root, specs['path'])
        with open(filepath, "w") as fileobj:
            yaml.dump(data, fileobj, Dumper=yaml.CDumper,
                      default_flow_style=False)
    elif filepath.endswith(".conf"):
        parser = RawConfigParser()
        parser.read(filepath)
        for section in parser.sections():
            path = parser.get(section, "specfile")
            parser.remove_option(section, "specfile")
            parser.set(section, "specfile", op.join(root, path))
        with open(filepath, "w") as fileobj:
            parser.write(fileobj)


class BaseTestCase(unittest.TestCase):

    """Base test class, introduces commonly required methods."""

    def assertKwargsEqual(self, dict1, dict2):
        """Assert that dictionaries are equal, to a deeper extent."""
        self.assertEqual(len(dict1.keys()), len(dict2.keys()))
        for key, value in dict1.iteritems():
            self.assertIn(key, dict2)
            left = value
            right = dict2[key]
            if isinstance(left, (tuple, list)):
                self.assertItemsEqual(left, right)
            elif isinstance(left, dict):
                self.assertDictEqual(left, right)
            else:
                self.assertEqual(left, right)

    def assertKwargsEmpty(self, data):
        """Assert that a dictionary is empty."""
        for value in data.itervalues():
            self.assertIn(value, ("", 0, 1, [], (), {}, None, False))

    def assertDataFrameEqual(self, dframe1, dframe2):
        """Assert that two dataframes are equal by their columns, indices and
        values."""
        self.assertTrue(np.all(dframe1.index == dframe2.index))
        self.assertTrue(np.all(dframe1.columns == dframe2.columns))
        for col in dframe1:
            try:
                self.assertTrue(np.all(dframe1[col] == dframe2[col]))
            except AssertionError:
                from IPython.core.debugger import Tracer
                Tracer()()
            self.assertEqual(dframe1[col].dtype, dframe2[col].dtype)

    def assertSeriesEqual(self, s1, s2):
        """Assert that two series are equal by their indices and values."""
        self.assertEqual(s1.shape, s2.shape)
        self.assertTrue(np.all(s1.values == s2.values))
        self.assertTrue(np.all(s1.index == s2.index))


def _dummy_converter(series):
    """Dummy function for testing converters on series objects."""
    return pd.Series([0 if "v" in i else 1 for i in series])


class BaseProjectTestCase(BaseTestCase):

    """Base class for tests of the Project module."""

    @classmethod
    def setUpClass(cls):
        cls.maxDiff = None
        # modify the testdata dict to have absolute paths
        with open(TEST_DATA_DICT, "r") as fileobj:
            test_data = yaml.load(fileobj, Loader=yaml.CLoader)
        for _, specs in test_data.iteritems():
            path = op.join(op.abspath(op.dirname(__file__)), specs['path'])
            specs['path'] = path
        # Put in the multifile specs
        cls.copied_iris_path = test_data['iris']['path'].replace("iris",
                                                                "iris2")
        dframe = pd.read_csv(test_data['iris']['path'])
        dframe.to_csv(cls.copied_iris_path, index=False)

        copied_iris_specs = deepcopy(test_data['iris'])
        copied_iris_specs['path'] = [copied_iris_specs['path'],
                                     cls.copied_iris_path]
        copied_iris_specs['nrows'] = [150, 150]
        test_data['multi_iris'] = copied_iris_specs

        with open(TEST_DATA_DICT, "w") as fileobj:
            yaml.dump(test_data, fileobj, Dumper=yaml.CDumper,
                      default_flow_style=False)
        cls.data_specs = test_data

        # Fix config file to have absolute paths

        config_fname = op.basename(TEST_CONFIG_FILE_PATH)
        cls.test_conf_file = op.join(os.getcwd(), config_fname)
        parser = RawConfigParser()
        parser.read(TEST_CONFIG_FILE_PATH)
        specfile = parser.get('pysemantic', 'specfile')
        specfile = op.join(op.abspath(op.dirname(__file__)), specfile)
        parser.remove_option("pysemantic", "specfile")
        parser.set("pysemantic", "specfile", specfile)
        with open(cls.test_conf_file, 'w') as fileobj:
            parser.write(fileobj)
        pr.CONF_FILE_NAME = config_fname

    @classmethod
    def tearDownClass(cls):
        try:
            # modify the testdata back
            with open(TEST_DATA_DICT, "r") as fileobj:
                test_data = yaml.load(fileobj, Loader=yaml.CLoader)
            test_data['iris']['path'] = op.join("testdata", "iris.csv")
            test_data['person_activity']['path'] = op.join("testdata",
                                                         "person_activity.tsv")
            del test_data['multi_iris']
            with open(TEST_DATA_DICT, "w") as fileobj:
                test_data = yaml.dump(test_data, fileobj, Dumper=yaml.CDumper,
                                     default_flow_style=False)

            # Change the config files back
            parser = RawConfigParser()
            parser.read(cls.test_conf_file)
            parser.remove_option("pysemantic", "specfile")
            parser.set("pysemantic", "specfile",
                       op.join("testdata", "test_dictionary.yaml"))
            with open(TEST_CONFIG_FILE_PATH, 'w') as fileobj:
                parser.write(fileobj)

        finally:
            os.unlink(cls.test_conf_file)
            os.unlink(cls.copied_iris_path)

    def setUp(self):
        iris_specs = {'sep': ',', 'dtype': {'Petal Length': float,
                                            'Sepal Width': float,
                                            'Petal Width': float,
                                            'Sepal Length': float,
                                            'Species': str},
                      'nrows': 150,
                      'filepath_or_buffer': op.join(
                                              op.abspath(op.dirname(__file__)),
                                              "testdata", "iris.csv")}
        copied_iris_specs = deepcopy(iris_specs)
        copied_iris_specs.update(
               {'filepath_or_buffer': iris_specs['filepath_or_buffer'].replace(
                                                        "iris", "iris2")})
        multi_iris_specs = [iris_specs, copied_iris_specs]
        person_activity_specs = {'sep': '\t', 'dtype': {'activity': str,
                                                        'sequence_name': str,
                                                        'tag': str, 'x': float,
                                                        'y': float, 'z': float,
                                                        },
                                 'parse_dates': ['date'], 'nrows': 100,
                                 'filepath_or_buffer': op.join(
                                              op.abspath(op.dirname(__file__)),
                                              "testdata",
                                              "person_activity.tsv")}
        expected = {'iris': iris_specs,
                    'person_activity': person_activity_specs,
                    'multi_iris': multi_iris_specs}
        self.expected_specs = expected
        self.project = pr.Project(project_name="pysemantic")


class TestConfig(BaseTestCase):

    """Test the configuration management utilities."""

    @classmethod
    def setUpClass(cls):
        # Fix the relative paths in the conig file.
        parser = RawConfigParser()
        parser.read(TEST_CONFIG_FILE_PATH)
        cls.old_fpath = parser.get("pysemantic", "specfile")
        parser.set("pysemantic", "specfile", op.abspath(cls.old_fpath))
        with open(TEST_CONFIG_FILE_PATH, "w") as fileobj:
            parser.write(fileobj)
        cls._parser = parser
        pr.CONF_FILE_NAME = "test.conf"

    @classmethod
    def tearDownClass(cls):
        cls._parser.set("pysemantic", "specfile", cls.old_fpath)
        with open(TEST_CONFIG_FILE_PATH, "w") as fileobj:
            cls._parser.write(fileobj)

    def setUp(self):
        self.testParser = RawConfigParser()
        for section in self._parser.sections():
            self.testParser.add_section(section)
            for item in self._parser.items(section):
                self.testParser.set(section, item[0], item[1])

    def test_loader_default_location(self):
        """Test if the config looks for the files in the correct places."""
        # Put the test config file in the current and home directories, with
        # some modifications.
        cwd_file = op.join(os.getcwd(), "test.conf")
        home_file = op.join(op.expanduser('~'), "test.conf")

        try:
            self.testParser.set("pysemantic", "specfile", os.getcwd())
            with open(cwd_file, "w") as fileobj:
                self.testParser.write(fileobj)
            specfile = pr.get_default_specfile("pysemantic")
            self.assertEqual(specfile, os.getcwd())

            os.unlink(cwd_file)

            self.testParser.set("pysemantic", "specfile", op.expanduser('~'))
            with open(home_file, "w") as fileobj:
                self.testParser.write(fileobj)
            specfile = pr.get_default_specfile("pysemantic")
            self.assertEqual(specfile, op.expanduser('~'))

        finally:
            os.unlink(home_file)


def _get_iris_args():
    """Get the ideal parser arguments for the iris dataset."""
    filepath = op.join(op.dirname(__file__), "testdata", "iris.csv")
    return dict(filepath_or_buffer=op.abspath(filepath),
                sep=",", nrows=150,
                dtype={'Petal Length': float,
                       'Petal Width': float,
                       'Sepal Length': float,
                       'Sepal Width': float,
                       'Species': str})


def _get_person_activity_args():
    """Get the ideal parser arguments for the activity dataset."""
    filepath = op.join(op.dirname(__file__), "testdata", "person_activity.tsv")
    return dict(filepath_or_buffer=op.abspath(filepath),
                sep="\t", nrows=100, dtype={'sequence_name': str,
                                            'tag': str,
                                            'x': float,
                                            'y': float,
                                            'z': float,
                                            'activity': str},
                parse_dates=['date'])

if __name__ == '__main__':
    unittest.main()
