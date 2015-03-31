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

import os
import unittest
import datetime
import tempfile
import warnings
import shutil
import subprocess
import os.path as op
from copy import deepcopy
from ConfigParser import RawConfigParser, NoSectionError

import yaml
import numpy as np
import pandas as pd
from pandas.io.parsers import ParserWarning
from traits.api import HasTraits, TraitError, Str, Type, List, Either

from validator import SchemaValidator, SeriesValidator, DataFrameValidator
import project as pr
from errors import MissingProject
from custom_traits import AbsFile, NaturalNumber, DTypesDict, ValidTraitList

TEST_CONFIG_FILE_PATH = op.join(op.abspath(op.dirname(__file__)), "testdata",
                                "test.conf")
TEST_DATA_DICT = op.join(op.abspath(op.dirname(__file__)), "testdata",
                         "test_dictionary.yaml")


def _path_fixer(filepath, root=None):
    """_path_fixer Change all the relative paths in `filepath` to absolute
    ones.

    :param filepath: File to be changed
    :param root: Root path with which the relative paths are prefixed. If None
    (default), the directory with this file is the root.
    """
    if root is None:
        root = op.join(op.abspath(op.dirname(__file__)))
    if filepath.endswith((".yaml", ".yml")):
        with open(filepath, "r") as f:
            data = yaml.load(f, Loader=yaml.CLoader)
        for specs in data.itervalues():
            specs['path'] = op.join(root, specs['path'])
        with open(filepath, "w") as f:
            data = yaml.dump(f, Dumper=yaml.CDumper)
    elif filepath.endswith(".conf"):
        parser = RawConfigParser()
        parser.read(filepath)
        for section in parser.sections():
            path = parser.get(section, "specfile")
            parser.remove_option(section, "specfile")
            parser.set(section, "specfile", op.join(root, path))
        with open(filepath, "w") as f:
            parser.write(f)


class BaseTestCase(unittest.TestCase):

    def assertKwargsEqual(self, dict1, dict2):
        """Assert that dictionaries are equal, to a deeper extent."""
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

    def assertKwargsEmpty(self, data):
        """Assert that a dictionary is empty."""
        for value in data.itervalues():
            self.assertIn(value, ("", 0, 1, [], (), {}, None, False))

    def assertDataFrameEqual(self, df1, df2):
        self.assertTrue(np.all(df1.index == df2.index))
        self.assertTrue(np.all(df1.columns == df2.columns))
        for col in df1:
            self.assertTrue(np.all(df1[col].values == df2[col].values))
            self.assertEqual(df1[col].dtype, df2[col].dtype)

    def assertSeriesEqual(self, s1, s2):
        self.assertEqual(s1.shape, s2.shape)
        self.assertTrue(np.all(s1.values == s2.values))
        self.assertTrue(np.all(s1.index == s2.index))


def _dummy_converter(series):
    return pd.Series([0 if "v" in i else 1 for i in series])


class TestSeriesValidator(BaseTestCase):
    """Tests for the SeriesValidator class."""

    @classmethod
    def setUpClass(cls):
        cls.dataframe = pd.read_csv(op.join(op.abspath(op.dirname(__file__)),
                                            "testdata", "iris.csv"))
        species_rules = {'unique_values': ['setosa', 'virginica',
                                           'versicolor'],
                         'drop_duplicates': False, 'drop_na': False}
        cls.species_rules = species_rules
        sepal_length_rules = {'drop_duplicates': False}
        cls.sepal_length_rules = sepal_length_rules

    def setUp(self):
        self.species = self.dataframe['Species'].copy()
        self.sepal_length = self.dataframe['Sepal Length'].copy()

    def test_unique_values(self):
        """Check if the series validator correctly checks for the unique
        values in the series."""
        validator = SeriesValidator(data=self.species,
                                    rules=self.species_rules)
        cleaned = validator.clean()
        self.assertItemsEqual(cleaned.unique(),
                              self.dataframe['Species'].unique())

    def test_bad_unique_values(self):
        """Check if the series validator drops values not specified in the
        species_rules."""
        # Add some bogus values
        noise = np.random.choice(['lily', 'petunia'], size=(50,))
        species = np.hstack((self.species.values, noise))
        np.random.shuffle(species)
        species = pd.Series(species)

        validator = SeriesValidator(data=species, rules=self.species_rules)
        cleaned = validator.clean()
        self.assertItemsEqual(cleaned.unique(),
                              self.dataframe['Species'].unique())

    def test_converter(self):
        """Test if the SeriesValidator properly applies converters."""
        self.species_rules['converters'] = [_dummy_converter]
        try:
            validator = SeriesValidator(data=self.species,
                                        rules=self.species_rules)
            cleaned = validator.clean()
            cleaned = cleaned.astype(bool)
            filtered = self.species[cleaned]
            self.assertEqual(filtered.nunique(), 1)
            self.assertItemsEqual(filtered.unique(), ['setosa'])
        finally:
            del self.species_rules['converters']

    def test_drop_duplicates(self):
        """Check if the SeriesValidator drops duplicates in the series."""
        self.species_rules['drop_duplicates'] = True
        try:
            x = self.species.unique().tolist()
            validator = SeriesValidator(data=self.species,
                                        rules=self.species_rules)
            cleaned = validator.clean()
            self.assertEqual(cleaned.shape[0], 3)
            self.assertItemsEqual(cleaned.tolist(), x)
        finally:
            self.species_rules['drop_duplicates'] = False

    def test_drop_na(self):
        """Check if the SeriesValidator drops NAs in the series."""
        self.species_rules['drop_na'] = True
        try:
            un = np.random.choice(self.species.unique().tolist() + [np.nan],
                                  size=(100,))
            un = pd.Series(un)
            validator = SeriesValidator(data=un,
                                        rules=self.species_rules)
            cleaned = validator.clean()
            self.assertEqual(cleaned.nunique(), self.species.nunique())
            self.assertItemsEqual(cleaned.unique().tolist(),
                                  self.species.unique().tolist())
        finally:
            self.species_rules['drop_na'] = False

    def test_numerical_series(self):
        """Test if the SeriesValidator works on a numerical series."""
        validator = SeriesValidator(data=self.sepal_length,
                                    rules=self.sepal_length_rules)
        cleaned = validator.clean()
        self.assertSeriesEqual(cleaned, self.dataframe['Sepal Length'])

    def test_min_max_rules(self):
        self.sepal_length_rules['min'] = 5.0
        self.sepal_length_rules['max'] = 7.0
        try:
            validator = SeriesValidator(data=self.sepal_length,
                                       rules=self.sepal_length_rules)
            cleaned = validator.clean()
            self.assertLessEqual(cleaned.max(), 7.0)
            self.assertGreaterEqual(cleaned.min(), 5.0)
        finally:
            del self.sepal_length_rules['max']
            del self.sepal_length_rules['min']

    def test_regex_filter(self):
        """Test if the SeriesValidator does filtering based on the regular
        expression provided."""
        self.species_rules['regex'] = r'\b[a-z]+\b'
        try:
            validator = SeriesValidator(data=self.species,
                                        rules=self.species_rules)
            cleaned = validator.clean()
            self.assertSeriesEqual(cleaned, self.dataframe['Species'])

            self.species = self.dataframe['Species'].copy()
            self.species = self.species.apply(lambda x: x.replace("e", "1"))
            validator = SeriesValidator(data=self.species,
                                        rules=self.species_rules)
            cleaned = validator.clean()
            self.assertItemsEqual(cleaned.shape, (50,))
            self.assertItemsEqual(cleaned.unique().tolist(), ['virginica'])
        finally:
            del self.species_rules['regex']


class TestDataFrameValidator(BaseTestCase):

    @classmethod
    def setUpClass(cls):
        cls.maxDiff = None
        with open(TEST_DATA_DICT, 'r') as f:
            basespecs = yaml.load(f, Loader=yaml.CLoader)
        # Fix the paths in basespecs
        for name, specs in basespecs.iteritems():
            rlpth = specs['path']
            specs['path'] = op.join(op.abspath(op.dirname(__file__)),
                                    rlpth)
        cls._basespecs = basespecs

        iris_validator = SchemaValidator(specification=cls._basespecs['iris'])
        pa_validator = SchemaValidator(
                               specification=cls._basespecs['person_activity'])
        iris_df = pd.read_csv(**iris_validator.get_parser_args())
        pa_df = pd.read_csv(**pa_validator.get_parser_args())
        cls.iris_df = iris_df
        cls.pa_df = pa_df

    def setUp(self):
        self.basespecs = deepcopy(self._basespecs)

    def test_column_rules(self):
        """Test if the DataFrame validator reads and enforces the column rules
        properly."""
        df_val = DataFrameValidator(data=self.iris_df.copy(),
                           column_rules=self.basespecs['iris']['column_rules'])
        cleaned = df_val.clean()
        self.assertDataFrameEqual(cleaned, self.iris_df.drop_duplicates())
        df_val = DataFrameValidator(data=self.pa_df.copy(),
                column_rules=self.basespecs['person_activity']['column_rules'])
        cleaned = df_val.clean()
        self.assertDataFrameEqual(cleaned, self.pa_df.drop_duplicates())

    def test_drop_duplicates(self):
        """Test if the DataFrameValidator is dropping duplicates properly."""
        col_rules = self.basespecs['iris'].get('column_rules')
        df = self.iris_df.copy()
        _df = pd.concat((df, df))
        validator = DataFrameValidator(data=_df, column_rules=col_rules)
        cleaned = validator.clean()
        self.assertDataFrameEqual(cleaned, df.drop_duplicates())


class TestProject(BaseTestCase):
    """ Tests for the project manager."""

    @classmethod
    def setUpClass(cls):
        cls.maxDiff = None
        # modify the testdata dict to have absolute paths
        with open(TEST_DATA_DICT, "r") as f:
            testData = yaml.load(f, Loader=yaml.CLoader)
        for name, specs in testData.iteritems():
            path = op.join(op.abspath(op.dirname(__file__)), specs['path'])
            specs['path'] = path
        # Put in the multifile specs
        cls.copied_iris_path = testData['iris']['path'].replace("iris",
                                                                "iris2")
        df = pd.read_csv(testData['iris']['path'])
        df.to_csv(cls.copied_iris_path, index=False)

        copied_iris_specs = deepcopy(testData['iris'])
        copied_iris_specs['path'] = [copied_iris_specs['path'],
                                     cls.copied_iris_path]
        copied_iris_specs['nrows'] = [150, 150]
        testData['multi_iris'] = copied_iris_specs

        with open(TEST_DATA_DICT, "w") as f:
            yaml.dump(testData, f, Dumper=yaml.CDumper,
                      default_flow_style=False)
        cls.data_specs = testData

        # Fix config file to have absolute paths

        config_fname = op.basename(TEST_CONFIG_FILE_PATH)
        cls.test_conf_file = op.join(os.getcwd(), config_fname)
        parser = RawConfigParser()
        parser.read(TEST_CONFIG_FILE_PATH)
        specfile = parser.get('pysemantic', 'specfile')
        specfile = op.join(op.abspath(op.dirname(__file__)), specfile)
        parser.remove_option("pysemantic", "specfile")
        parser.set("pysemantic", "specfile", specfile)
        with open(cls.test_conf_file, 'w') as f:
            parser.write(f)
        pr.CONF_FILE_NAME = config_fname

    @classmethod
    def tearDownClass(cls):
        try:
            # modify the testdata back
            with open(TEST_DATA_DICT, "r") as f:
                testData = yaml.load(f, Loader=yaml.CLoader)
            testData['iris']['path'] = op.join("testdata", "iris.csv")
            testData['person_activity']['path'] = op.join("testdata",
                                                         "person_activity.tsv")
            del testData['multi_iris']
            with open(TEST_DATA_DICT, "w") as f:
                testData = yaml.dump(testData, f, Dumper=yaml.CDumper,
                                     default_flow_style=False)

            # Change the config files back
            parser = RawConfigParser()
            parser.read(cls.test_conf_file)
            parser.remove_option("pysemantic", "specfile")
            parser.set("pysemantic", "specfile",
                       op.join("testdata", "test_dictionary.yaml"))
            with open(TEST_CONFIG_FILE_PATH, 'w') as f:
                parser.write(f)

        finally:
            os.unlink(cls.test_conf_file)
            os.unlink(cls.copied_iris_path)

    def setUp(self):
        iris_specs = {'sep': ',',
                             'dtype': {'Petal Length': float,
                                       'Sepal Width': float,
                                       'Petal Width': float,
                                       'Sepal Length': float,
                                       'Species': str},
                             'usecols': ['Petal Length', 'Sepal Length',
                                         'Sepal Width', 'Petal Width',
                                         'Species'],
                             'nrows': 150,
                             'filepath_or_buffer': op.join(
                                              op.abspath(op.dirname(__file__)),
                                              "testdata", "iris.csv")}
        copied_iris_specs = deepcopy(iris_specs)
        copied_iris_specs.update(
               {'filepath_or_buffer': iris_specs['filepath_or_buffer'].replace(
                                                        "iris", "iris2")})
        multi_iris_specs = [iris_specs, copied_iris_specs]
        person_activity_specs = {'sep': '\t',
                                'dtype': {'activity': str,
                                          'sequence_name': str,
                                          'tag': str, 'x': float,
                                          'y': float, 'z': float,
                                          },
                                'usecols': ['activity',
                                            'sequence_name', 'tag',
                                            'x', 'y', 'z', 'date'],
                                'parse_dates': ['date'],
                                'nrows': 100,
                                'filepath_or_buffer': op.join(
                                      op.abspath(op.dirname(__file__)),
                                      "testdata",
                                      "person_activity.tsv")}
        expected = {'iris': iris_specs,
                    'person_activity': person_activity_specs,
                    'multi_iris': multi_iris_specs}
        self.expected_specs = expected
        self.project = pr.Project(project_name="pysemantic")

    def test_add_dataset(self):
        """Test if adding datasets programmatically works fine."""
        tempdir = tempfile.mkdtemp()
        outpath = op.join(tempdir, "foo.csv")
        df = pd.DataFrame(np.random.random((10, 10)))
        df.to_csv(outpath, index=False)
        specs = dict(path=outpath, delimiter=',', ncols=10, nrows=10)
        try:
            pr.add_dataset("pysemantic", "sample_dataset", specs)
            parsed_specs = pr.get_schema_specs("pysemantic", "sample_dataset")
            self.assertKwargsEqual(specs, parsed_specs)
        finally:
            shutil.rmtree(tempdir)
            with open(TEST_DATA_DICT, "r") as f:
                test_specs = yaml.load(f, Loader=yaml.CLoader)
            del test_specs['sample_dataset']
            with open(TEST_DATA_DICT, "w") as f:
                yaml.dump(test_specs, f, Dumper=yaml.CDumper,
                          default_flow_style=False)

    def test_remove_dataset(self):
        """Test if programmatically removing a dataset works."""
        with open(TEST_DATA_DICT, "r") as f:
            specs = yaml.load(f, Loader=yaml.CLoader)
        try:
            pr.remove_dataset("pysemantic", "iris")
            self.assertRaises(KeyError, pr.get_schema_specs, "pysemantic",
                              "iris")
        finally:
            with open(TEST_DATA_DICT, "w") as f:
                yaml.dump(specs, f, Dumper=yaml.CDumper,
                          default_flow_style=False)

    def test_regex_separator(self):
        """Test if the project properly loads a dataset when it encounters
        regex separators."""
        tempdir = tempfile.mkdtemp()
        outfile = op.join(tempdir, "sample.txt")
        data = ["col1"] + map(str, range(10))
        with open(outfile, "w") as f:
            f.write("\n".join(data))
        specs = dict(path=outfile, delimiter=r'\n', dtypes={'col1': int})
        pr.add_dataset("pysemantic", "sample_dataset", specs)
        try:
            _pr = pr.Project("pysemantic")
            with warnings.catch_warnings(record=True) as catcher:
                df = _pr.load_dataset("sample_dataset")
                assert len(catcher) == 2
                assert issubclass(catcher[1].category, ParserWarning)
            data.remove("col1")
            self.assertItemsEqual(map(int, data), df['col1'].tolist())
        finally:
            pr.remove_dataset("pysemantic", "sample_dataset")
            shutil.rmtree(tempdir)

    def test_get_schema_spec(self):
        """test the module level function to get schema specifications"""
        specs = pr.get_schema_specs("pysemantic")
        self.assertKwargsEqual(specs, self.data_specs)

    def test_load_dataset_wrong_dtypes_in_spec(self):
        """Test if the loader can safely load columns that have a wrongly
        specified data type in the schema."""
        # Make a file with two columns, both specified as integers in the
        # dtypes, but one has random string types.
        x = np.random.randint(0, 10, size=(100, 2))
        df = pd.DataFrame(x, columns=['a', 'b'])
        tempdir = tempfile.mkdtemp()
        outfile = op.join(tempdir, "testdata.csv")
        _ix = np.random.randint(0, 100, size=(5,))
        df['b'][_ix] = "aa"
        df.to_csv(outfile, index=False)
        specs = dict(delimiter=',', dtypes={'a': int, 'b': int}, path=outfile)
        specfile = op.join(tempdir, "dict.yaml")
        with open(specfile, "w") as f:
            yaml.dump({'testdata': specs}, f, Dumper=yaml.CDumper)
        pr.add_project("wrong_dtype", specfile)
        try:
            _pr = pr.Project("wrong_dtype")
            with warnings.catch_warnings(record=True) as catcher:
                df = _pr.load_dataset("testdata")
                assert len(catcher) == 1
                assert issubclass(catcher[-1].category, UserWarning)
        finally:
            pr.remove_project("wrong_dtype")
            shutil.rmtree(tempdir)

    def test_load_dataset_missing_nrows(self):
        """Test if the project loads datasets properly if the nrows parameter
        is not provided in the schema."""
        # Modify the schema to remove the nrows
        with open(TEST_DATA_DICT, "r") as f:
            org_specs = yaml.load(f, Loader=yaml.CLoader)
        new_specs = deepcopy(org_specs)
        for dataset_specs in new_specs.itervalues():
            del dataset_specs['nrows']
        with open(TEST_DATA_DICT, "w") as f:
            yaml.dump(new_specs, f, Dumper=yaml.CDumper)
        try:
            _pr = pr.Project("pysemantic")
            df = pd.read_csv(**self.expected_specs['iris'])
            loaded = _pr.load_dataset("iris")
            self.assertDataFrameEqual(df, loaded)
            df = pd.read_table(**self.expected_specs['person_activity'])
            loaded = _pr.load_dataset("person_activity")
            self.assertDataFrameEqual(loaded, df)
        finally:
            with open(TEST_DATA_DICT, "w") as f:
                yaml.dump(org_specs, f, Dumper=yaml.CDumper)

    def test_set_schema_fpath(self):
        """Test if programmatically setting a schema file to an existing
        project works."""
        old_schempath = pr._get_default_specfile("pysemantic")
        try:
            self.assertTrue(pr.set_schema_fpath("pysemantic", "/foo/bar"))
            self.assertEqual(pr._get_default_specfile("pysemantic"),
                             "/foo/bar")
            self.assertRaises(MissingProject, pr.set_schema_fpath,
                              "foobar", "/foo/bar")
        finally:
            conf_path = pr._locate_config_file()
            parser = RawConfigParser()
            parser.read(conf_path)
            parser.remove_option("pysemantic", "specfile")
            parser.set("pysemantic", "specfile", old_schempath)
            with open(TEST_CONFIG_FILE_PATH, "w") as f:
                parser.write(f)

    def test_get_project_specs(self):
        """Check if the project manager produces all specifications correctly.
        """
        specs = self.project.get_project_specs()
        for name, argdict in specs.iteritems():
            if isinstance(argdict, list):
                for i in range(len(argdict)):
                    self.assertKwargsEqual(argdict[i],
                                           self.expected_specs[name][i])
            else:
                self.assertKwargsEqual(argdict, self.expected_specs[name])

    def test_get_dataset_specs(self):
        """Check if the project manager produces specifications for each
        dataset correctly."""
        for name in ['iris', 'person_activity']:
            self.assertKwargsEqual(self.project.get_dataset_specs(name),
                                   self.expected_specs[name])

    def test_parser(self):
        """Check if the dataset assigns the correct parser to the loader."""
        iris_specs = self.project.get_dataset_specs("iris")
        self.project._update_parser(iris_specs)
        self.assertEqual(self.project.parser, pd.read_csv)
        person_specs = self.project.get_dataset_specs("person_activity")
        self.project._update_parser(person_specs)
        self.assertEqual(self.project.parser, pd.read_table)

    def test_get_multifile_dataset_specs(self):
        outArgs = self.project.get_dataset_specs("multi_iris")
        self.assertTrue(isinstance(outArgs, list))
        self.assertEqual(len(outArgs), len(self.expected_specs['multi_iris']))
        for i in range(len(outArgs)):
            self.assertKwargsEqual(outArgs[i],
                                   self.expected_specs['multi_iris'][i])

    def test_set_dataset_specs(self):
        """Check if setting dataset specifications through the Project object
        works."""
        path = op.join(op.abspath(op.dirname(__file__)), "testdata",
                       "iris.csv")
        specs = dict(filepath_or_buffer=path,
                     usecols=['Sepal Length', 'Petal Width', 'Species'],
                     dtype={'Sepal Length': str})
        self.assertTrue(self.project.set_dataset_specs("iris", specs))
        expected = pd.read_csv(**specs)
        loaded = self.project.load_dataset("iris")
        self.assertDataFrameEqual(expected, loaded)

    def test_set_dataset_specs_to_file(self):
        """Check if newly set dataset specifications are written to file
        properly."""
        try:
            with open(TEST_DATA_DICT, "r") as f:
                oldspecs = yaml.load(f, Loader=yaml.CLoader)
            path = op.join(op.abspath(op.dirname(__file__)), "testdata",
                           "iris.csv")
            specs = dict(filepath_or_buffer=path,
                         usecols=['Sepal Length', 'Petal Width', 'Species'],
                         dtype={'Sepal Length': str})
            self.assertTrue(self.project.set_dataset_specs("iris", specs,
                                                           write_to_file=True))
            with open(TEST_DATA_DICT, "r") as f:
                newspecs = yaml.load(f, Loader=yaml.CLoader)
            self.assertKwargsEqual(newspecs['iris'], specs)
        finally:
            with open(TEST_DATA_DICT, "w") as f:
                yaml.dump(oldspecs, f, Dumper=yaml.CDumper)

    def test_add_project(self):
        """Test if adding a project works properly."""
        test_project_name = "test_project"
        pr.add_project(test_project_name, TEST_DATA_DICT)
        # Check if the project name is indeed present in the config file
        test_dict = pr._get_default_specfile(test_project_name)
        self.assertTrue(test_dict, TEST_DATA_DICT)

    def test_remove_project(self):
        """Test if removing a project works properly."""
        self.assertTrue(pr.remove_project("test_project"))
        self.assertRaises(NoSectionError, pr._get_default_specfile,
                          "test_project")

    def test_load_all(self):
        """Test if loading all datasets in a project works as expected."""
        loaded = self.project.load_datasets()
        self.assertItemsEqual(loaded.keys(), ('iris', 'person_activity',
                                              'multi_iris'))
        df = pd.read_csv(**self.expected_specs['iris'])
        self.assertDataFrameEqual(loaded['iris'], df)
        df = pd.read_csv(**self.expected_specs['person_activity'])
        self.assertDataFrameEqual(loaded['person_activity'], df)
        dfs = [pd.read_csv(**args) for args in
               self.expected_specs['multi_iris']]
        dfs = [x.drop_duplicates() for x in dfs]
        df = pd.concat(dfs)
        self.assertDataFrameEqual(loaded['multi_iris'], df)

    def test_dataset_colnames(self):
        """Check if the column names read by the loader are correct."""
        for name in ['iris', 'person_activity']:
            loaded = self.project.load_dataset(name)
            columns = loaded.columns.tolist()
            spec_colnames = self.data_specs[name]['dtypes'].keys()
            self.assertItemsEqual(spec_colnames, columns)

    def test_dataset_coltypes(self):
        """Check whether the columns have the correct datatypes."""
        for name in ['iris', 'person_activity']:
            loaded = self.project.load_dataset(name)
            for colname in loaded:
                if loaded[colname].dtype == np.dtype('O'):
                    self.assertEqual(self.data_specs[name]['dtypes'][colname],
                                     str)
                elif loaded[colname].dtype == np.dtype('<M8[ns]'):
                    self.assertEqual(self.data_specs[name]['dtypes'][colname],
                                     datetime.date)
                else:
                    self.assertEqual(loaded[colname].dtype,
                                     self.data_specs[name]['dtypes'][colname])


class TestCLI(BaseTestCase):
    """Test the pysemantic CLI."""

    @classmethod
    def setUpClass(cls):
        os.environ['PYSEMANTIC_CONFIG'] = "test.conf"
        pr.CONF_FILE_NAME = "test.conf"
        cls.testenv = os.environ
        cls.test_config_path = op.join(os.getcwd(), "test.conf")
        shutil.copy(TEST_CONFIG_FILE_PATH, cls.test_config_path)
        # Change the relative paths in the config file to absolute paths
        parser = RawConfigParser()
        parser.read(cls.test_config_path)
        for section in parser.sections():
            schema_path = parser.get(section, "specfile")
            parser.remove_option(section, "specfile")
            parser.set(section, "specfile",
                       op.join(op.abspath(op.dirname(__file__)), schema_path))
        with open(cls.test_config_path, "w") as f:
            parser.write(f)
        # change the relative paths in the test dictionary to absolute paths
        with open(TEST_DATA_DICT, "r") as f:
            cls.org_specs = yaml.load(f, Loader=yaml.CLoader)
        new_specs = deepcopy(cls.org_specs)
        for dataset_name, specs in new_specs.iteritems():
            path = specs['path']
            specs['path'] = op.join(op.abspath(op.dirname(__file__)), path)
        # Rewrite this to the file
        with open(TEST_DATA_DICT, "w") as f:
            yaml.dump(new_specs, f, Dumper=yaml.CDumper)

    @classmethod
    def tearDownClass(cls):
        os.unlink(cls.test_config_path)
        # Rewrite the original specs back to the config dir
        with open(TEST_DATA_DICT, "w") as f:
            yaml.dump(cls.org_specs, f, Dumper=yaml.CDumper,
                      default_flow_style=False)

    def setUp(self):
        pr.add_project("dummy_project", "/foo/bar.yaml")

    def tearDown(self):
        pr.remove_project("dummy_project")

    def test_set_specification(self):
        """Test if the set-specs subcommand of the CLI worls properly."""
        org_specs = pr.get_schema_specs("pysemantic")
        cmd = ['semantic', 'set-specs', 'pysemantic', '--dataset', 'iris',
               '--dlm', '|']
        try:
            subprocess.check_call(cmd, env=self.testenv)
            new_specs = pr.get_schema_specs("pysemantic", "iris")
            self.assertEqual(new_specs['delimiter'], '|')
        finally:
            for dataset_name, specs in org_specs.iteritems():
                pr.set_schema_specs("pysemantic", dataset_name, **specs)

    def test_list_projects(self):
        """Test if the `list` subcommand of the CLI works properly."""
        cmd = ['semantic', 'list']
        output = subprocess.check_output(cmd, env=self.testenv).splitlines()
        path = op.join(op.abspath(op.dirname(__file__)),
                       "testdata/test_dictionary.yaml")
        dummy_data = [("pysemantic", path),
                      ("dummy_project", "/foo/bar.yaml")]
        for i, config in enumerate(dummy_data):
            ideal = "Project {0} with specfile at {1}".format(*config)
            self.assertEqual(ideal, output[i])

    def test_add(self):
        """Test if the `add` subcommand can add projects to the config file."""
        try:
            cmd = ['semantic', 'add', 'dummy_added_project', '/tmp/dummy.yaml']
            subprocess.check_call(cmd, env=self.testenv)
            projects = pr.get_projects()
            self.assertIn(("dummy_added_project", "/tmp/dummy.yaml"), projects)
        finally:
            pr.remove_project("dummy_added_project")

    def test_add_dataset(self):
        """Test if the add-dataset subcommand adds datasets to projects
        correctly"""
        tempdir = tempfile.mkdtemp()
        outfile = op.join(tempdir, "testdata.csv")
        df = pd.DataFrame(np.random.random((10, 2)), columns=['a', 'b'])
        df.to_csv(outfile, index=False)
        cmd = ("semantic add-dataset testdata --project pysemantic --path {}"
               " --dlm ,")
        cmd = cmd.format(outfile).split(" ")
        try:
            subprocess.check_call(cmd, env=self.testenv)
            _pr = pr.Project("pysemantic")
            self.assertIn("testdata", _pr.datasets)
            specs = dict(path=outfile, delimiter=',')
            actual = pr.get_schema_specs("pysemantic", "testdata")
            self.assertKwargsEqual(specs, actual)
        finally:
            pr.remove_dataset("pysemantic", "testdata")
            shutil.rmtree(tempdir)

    def test_remove(self):
        """Test if the `remove` subcommand can remove projects from the config
        file."""
        pr.add_project("dummy_project_2", "/foo/baz.yaml")
        try:
            cmd = ['semantic', 'remove', 'dummy_project_2']
            subprocess.check_call(cmd, env=self.testenv)
            projects = pr.get_projects()
            proj_names = [p[0] for p in projects]
            self.assertNotIn("dummy_project_2", proj_names)
        finally:
            pr.remove_project("dummy_project_2")

    def test_remove_nonexistent_project(self):
        """Check if attempting to remove a nonexistent project fails."""
        cmd = ['semantic', 'remove', 'foobar']
        output = subprocess.check_output(cmd, env=self.testenv)
        self.assertEqual(output.strip(), "Removing the project foobar failed.")

    def test_set_schema(self):
        """Test if the set-schema subcommand works."""
        cmd = ['semantic', 'set-schema', 'dummy_project', '/tmp/baz.yaml']
        subprocess.check_call(cmd, env=self.testenv)
        self.assertEqual(pr._get_default_specfile('dummy_project'),
                         '/tmp/baz.yaml')

    def test_set_schema_nonexistent_project(self):
        """Test if the set-schema prints proper warnings when trying to set
        schema file for nonexistent project."""
        cmd = ['semantic', 'set-schema', 'dummy_project_3', '/foo']
        output = subprocess.check_output(cmd, env=self.testenv)
        msg = """Project {} not found in the configuration. Please use
            $ semantic add
            to register the project.""".format("dummy_project_3")
        self.assertEqual(output.strip(), msg)

    def test_relative_path(self):
        """Check if the set-schema and add subcommands convert relative paths
        from the cmdline to absolute paths in the config file."""
        try:
            cmd = ['semantic', 'set-schema', 'dummy_project', './foo.yaml']
            subprocess.check_call(cmd, env=self.testenv)
            self.assertTrue(op.isabs(pr._get_default_specfile(
                                                             'dummy_project')))
            pr.remove_project("dummy_project")
            cmd = ['semantic', 'add', 'dummy_project', './foo.yaml']
            subprocess.check_call(cmd, env=self.testenv)
            self.assertTrue(op.isabs(pr._get_default_specfile(
                                                             'dummy_project')))
        finally:
            pr.remove_project("dummy_project_1")


class TestConfig(BaseTestCase):
    """
    Test the configuration management utilities.
    """

    @classmethod
    def setUpClass(cls):
        # Fix the relative paths in the conig file.
        parser = RawConfigParser()
        parser.read(TEST_CONFIG_FILE_PATH)
        cls.old_fpath = parser.get("pysemantic", "specfile")
        parser.set("pysemantic", "specfile", op.abspath(cls.old_fpath))
        with open(TEST_CONFIG_FILE_PATH, "w") as f:
            parser.write(f)
        cls._parser = parser
        pr.CONF_FILE_NAME = "test.conf"

    @classmethod
    def tearDownClass(cls):
        cls._parser.set("pysemantic", "specfile", cls.old_fpath)
        with open(TEST_CONFIG_FILE_PATH, "w") as f:
            cls._parser.write(f)

    def setUp(self):
        self.testParser = RawConfigParser()
        for section in self._parser.sections():
            self.testParser.add_section(section)
            for item in self._parser.items(section):
                self.testParser.set(section, item[0], item[1])

    def test_config_loader_default_location(self):
        """Check if the config loader looks for the files in the correct
        places."""
        # Put the test config file in the current and home directories, with
        # some modifications.
        cwd_file = op.join(os.getcwd(), "test.conf")
        home_file = op.join(op.expanduser('~'), "test.conf")

        try:
            self.testParser.set("pysemantic", "specfile", os.getcwd())
            with open(cwd_file, "w") as f:
                self.testParser.write(f)
            specfile = pr._get_default_specfile("pysemantic")
            self.assertEqual(specfile, os.getcwd())

            os.unlink(cwd_file)

            self.testParser.set("pysemantic", "specfile", op.expanduser('~'))
            with open(home_file, "w") as f:
                self.testParser.write(f)
            specfile = pr._get_default_specfile("pysemantic")
            self.assertEqual(specfile, op.expanduser('~'))

        finally:
            os.unlink(home_file)


class TestSchemaValidator(BaseTestCase):
    """
    Test the `pysemantic.validator.SchemaValidatorClass`
    """

    @classmethod
    def setUpClass(cls):
        cls.maxDiff = None
        cls.specfile = op.join(op.dirname(__file__), "testdata",
                               "test_dictionary.yaml")
        with open(cls.specfile, "r") as f:
            cls._basespecs = yaml.load(f, Loader=yaml.CLoader)
        cls.specs = deepcopy(cls._basespecs)

        # fix the paths in basespecs if they aren't absolute
        for name, dataspec in cls.specs.iteritems():
            if not op.isabs(dataspec['path']):
                dataspec['path'] = op.join(op.abspath(op.dirname(__file__)),
                                           dataspec['path'])
        # The updated values also need to be dumped into the yaml file, because
        # some functionality of the validator depends on parsing it.
        with open(cls.specfile, "w") as f:
            yaml.dump(cls.specs, f, Dumper=yaml.CDumper,
                      default_flow_style=False)

        cls.ideal_activity_parser_args = _get_person_activity_args()
        cls.ideal_iris_parser_args = _get_iris_args()

    @classmethod
    def tearDownClass(cls):
        with open(cls.specfile, "w") as f:
            yaml.dump(cls._basespecs, f, Dumper=yaml.CDumper,
                      default_flow_style=False)

    def setUp(self):
        # FIXME: This should not be necessary, but without it, a couple of
        # tests strangely fail. I think one or both of the following two tests
        # are messing up the base specifications.
        self.basespecs = deepcopy(self.specs)

    def test_from_dict(self):
        """Test if the SchemaValidator.from_dict constructor works."""
        validator = SchemaValidator.from_dict(self.basespecs['iris'])
        self.assertKwargsEqual(validator.get_parser_args(),
                               self.ideal_iris_parser_args)
        validator = SchemaValidator.from_dict(self.basespecs[
                                                            'person_activity'])
        self.assertKwargsEqual(validator.get_parser_args(),
                               self.ideal_activity_parser_args)

    def test_from_specfile(self):
        """Test if the SchemaValidator.from_specfile constructor works."""
        validator = SchemaValidator.from_specfile(self.specfile, "iris")
        self.assertKwargsEqual(validator.get_parser_args(),
                               self.ideal_iris_parser_args)
        validator = SchemaValidator.from_specfile(self.specfile,
                                                  "person_activity")
        self.assertKwargsEqual(validator.get_parser_args(),
                               self.ideal_activity_parser_args)

    def test_to_dict(self):
        """Test if the SchemaValidator.to_dict method works."""
        validator = SchemaValidator(specification=self.basespecs['iris'])
        self.assertKwargsEqual(validator.to_dict(),
                               self.ideal_iris_parser_args)
        validator = SchemaValidator(specification=self.basespecs[
                                                            'person_activity'])
        self.assertKwargsEqual(validator.to_dict(),
                               self.ideal_activity_parser_args)

    def test_required_args(self):
        """Test if the required arguments for the validator are working
        properly."""
        filepath = self.basespecs['iris'].pop('path')
        delimiter = self.basespecs['iris'].pop('delimiter')
        try:
            # Remove the path and delimiter from the scehma
            self.assertRaises(TraitError, SchemaValidator,
                              specification=self.basespecs['iris'])
        finally:
            self.basespecs['iris']['delimiter'] = delimiter
            self.basespecs['iris']['path'] = filepath

    def test_multifile_dataset_schema(self):
        """Test if a dataset schema with multiple files works properly."""
        duplicate_iris_path = self.basespecs['iris']['path'].replace("iris",
                                                                     "iris2")
        # Copy the file
        df = pd.read_csv(self.basespecs['iris']['path'])
        df.to_csv(duplicate_iris_path, index=False)

        # Create the news chema
        schema = {'nrows': [150, 150], 'path': [duplicate_iris_path,
                  self.basespecs['iris']['path']]}
        for k, v in self.basespecs['iris'].iteritems():
            if k not in schema:
                schema[k] = v

        try:
            validator = SchemaValidator(specification=schema)
            self.assertTrue(validator.is_multifile)
            self.assertItemsEqual(validator.filepath, schema['path'])
            self.assertItemsEqual(validator.nrows, schema['nrows'])
            validated_args = validator.get_parser_args()
            self.assertTrue(isinstance(validated_args, list))
            self.assertEqual(len(validated_args), 2)
        finally:
            os.unlink(duplicate_iris_path)

    def test_validator_with_specdict_iris(self):
        """Check if the validator works when only the specification is supplied
        as a dictionary for the iris dataset."""
        validator = SchemaValidator(specification=self.basespecs['iris'])
        self.assertFalse(validator.is_multifile)
        validated_parser_args = validator.get_parser_args()
        self.assertKwargsEqual(validated_parser_args,
                               self.ideal_iris_parser_args)

    def test_validator_with_specfile_spec(self):
        """Check if the validator works when the specfile and specification are
        both provided."""
        # This is necessary because the validator might have to write
        # specifications to the dictionary.
        validator = SchemaValidator(specification=self.basespecs['iris'],
                                    specfile=self.specfile)
        self.assertFalse(validator.is_multifile)
        validated_parser_args = validator.get_parser_args()
        self.assertKwargsEqual(validated_parser_args,
                               self.ideal_iris_parser_args)

    def test_validator_with_specdist_activity(self):
        """Check if the validator works when only the specification is supplied
        as a dictionary for the person activity dataset."""
        validator = SchemaValidator(
                               specification=self.basespecs['person_activity'])
        self.assertFalse(validator.is_multifile)
        validated = validator.get_parser_args()
        self.assertKwargsEqual(validated, self.ideal_activity_parser_args)

    def test_error_for_relative_filepath(self):
        """Test if validator raises errors when relative paths are found in the
        dictionary."""
        specs = self.basespecs['iris']
        old_path = specs['path']
        try:
            specs['path'] = op.join("testdata", "iris.csv")
            self.assertRaises(TraitError, SchemaValidator,
                              specification=specs)
        finally:
            specs['path'] = old_path

    def test_error_for_bad_dtypes(self):
        """Check if the validator raises an error if a bad dtype dictionary is
        passed."""
        specs = self.basespecs['iris']
        old = specs['dtypes'].pop('Species')
        try:
            specs['dtypes']['Species'] = "random_string"
            validator = SchemaValidator(specification=specs)
            self.assertRaises(TraitError, validator.get_parser_args)
        finally:
            specs['dtypes']['Species'] = old

    def test_error_only_specfile(self):
        """Test if the validator fails when only the path to the specfile is
        provided. """
        self.assertRaises(TraitError, SchemaValidator, specfile=self.specfile)

    def test_error_only_name(self):
        """Test if the validator fails when only the path to the specfile is
        provided. """
        self.assertRaises(TraitError, SchemaValidator, name="iris")

    def test_validator_specfile_name_iris(self):
        """Test if the validator works when providing specifle and name for the
        iris dataset."""
        validator = SchemaValidator(specfile=self.specfile, name="iris")
        validated_parser_args = validator.get_parser_args()
        self.assertKwargsEqual(validated_parser_args,
                               self.ideal_iris_parser_args)

    def test_validator_specfile_name_activity(self):
        """Test if the validator works when providing specifle and name for the
        activity dataset."""
        validator = SchemaValidator(specfile=self.specfile,
                                    name="person_activity")
        validated_parser_args = validator.get_parser_args()
        self.assertKwargsEqual(validated_parser_args,
                               self.ideal_activity_parser_args)


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


class TestCustomTraits(unittest.TestCase):
    """ Testcase for the custom_traits module. This consists purely of testing
    whether validation is happening correctly on the custom_traits."""

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
        """Test if `pysemantic.self.traits.ValidTraitsList` works properly.
        """
        self.assertItemsEqual(self.traits.required, ['filepath', 'number',
                                                     'dtype'])

    def test_natural_number_either_list_trait(self):
        """Test of the NaturalNumber trait works within Either and List traits
        """
        self.traits.numberlist = 1
        self.traits.numberlist = [1, 2]
        self.assertRaises(TraitError, self.setter, "numberlist", 0)
        self.assertRaises(TraitError, self.setter, "numberlist", [0, 1])

    def test_absfile_either_list_traits(self):
        """Test if the AbsFile trait works within Either and List self.traits
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
