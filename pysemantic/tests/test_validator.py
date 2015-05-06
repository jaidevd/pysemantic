#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2015 jaidev <jaidev@newton>
#
# Distributed under terms of the MIT license.

"""Tests for the validator module."""

import os
import os.path as op
import unittest
import tempfile
import warnings
import shutil
from copy import deepcopy

import numpy as np
import pandas as pd
import yaml
from traits.api import TraitError

from pysemantic.tests.test_base import (BaseTestCase, _dummy_converter,
                                        TEST_DATA_DICT, _get_iris_args,
                                        _get_person_activity_args)
from pysemantic.validator import (SeriesValidator, SchemaValidator,
                                  DataFrameValidator)
from pysemantic.utils import get_md5_checksum

try:
    from yaml import CLoader as Loader
    from yaml import CDumper as Dumper
except ImportError:
    from yaml import Loader
    from yaml import Dumper


class TestSchemaValidator(BaseTestCase):

    """Test the `pysemantic.validator.SchemaValidatorClass`."""

    @classmethod
    def setUpClass(cls):
        cls.maxDiff = None
        cls.specfile = op.join(op.dirname(__file__), "testdata",
                               "test_dictionary.yaml")
        with open(cls.specfile, "r") as fileobj:
            cls._basespecs = yaml.load(fileobj, Loader=Loader)
        cls.specs = deepcopy(cls._basespecs)

        # fix the paths in basespecs if they aren't absolute
        for _, dataspec in cls.specs.iteritems():
            if not op.isabs(dataspec['path']):
                dataspec['path'] = op.join(op.abspath(op.dirname(__file__)),
                                           dataspec['path'])
        # The updated values also need to be dumped into the yaml file, because
        # some functionality of the validator depends on parsing it.
        with open(cls.specfile, "w") as fileobj:
            yaml.dump(cls.specs, fileobj, Dumper=Dumper,
                      default_flow_style=False)

        cls.ideal_activity_parser_args = _get_person_activity_args()
        cls.ideal_iris_parser_args = _get_iris_args()

    @classmethod
    def tearDownClass(cls):
        with open(cls.specfile, "w") as fileobj:
            yaml.dump(cls._basespecs, fileobj, Dumper=Dumper,
                      default_flow_style=False)

    def setUp(self):
        # FIXME: This should not be necessary, but without it, a couple of
        # tests strangely fail. I think one or both of the following two tests
        # are messing up the base specifications.
        self.basespecs = deepcopy(self.specs)

    def test_na_values(self):
        """Test if adding NA values in the schema works properly."""
        bad_iris_path = op.join(op.abspath(op.dirname(__file__)), "testdata",
                                "bad_iris.csv")
        schema = deepcopy(self.basespecs['iris'])
        schema['path'] = bad_iris_path
        schema['column_rules']['Species']['unique_values'].append('unknown')
        schema['column_rules']['Species']['na_values'] = ['unknown']
        validator = SchemaValidator(specification=schema)
        parser_args = validator.get_parser_args()
        self.assertDictEqual(parser_args.get("na_values"),
                             {'Species': ['unknown']})

    def test_md5(self):
        """Check if the md5 checksum validation works properly."""
        schema = deepcopy(self.basespecs["iris"])
        schema['md5'] = get_md5_checksum(schema['path'])
        SchemaValidator(specification=schema)
        tempdir = tempfile.mkdtemp()
        outpath = op.join(tempdir, "bad_iris.csv")
        iris = pd.read_csv(schema['path'])
        del iris['Species']
        iris.to_csv(outpath, index=False)
        schema['path'] = outpath
        try:
            with warnings.catch_warnings(record=True) as catcher:
                SchemaValidator(specification=schema).get_parser_args()
                assert len(catcher) == 1
                assert issubclass(catcher[-1].category, UserWarning)
        finally:
            shutil.rmtree(tempdir)

    def test_pandas_defaults_empty_specs(self):
        """Test if the validator falls back to pandas defaults for empty specs.
        """
        schema = dict(path=op.join(op.abspath(op.dirname(__file__)),
                                   "testdata", "iris.csv"))
        validator = SchemaValidator(specification=schema)
        ideal = pd.read_csv(schema['path'])
        actual = pd.read_csv(**validator.get_parser_args())
        self.assertDataFrameEqual(ideal, actual)

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
        properly.
        """
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
        dframe = pd.read_csv(self.basespecs['iris']['path'])
        dframe.to_csv(duplicate_iris_path, index=False)

        # Create the news chema
        schema = {'nrows': [150, 150], 'path': [duplicate_iris_path,
                  self.basespecs['iris']['path']]}
        for key, value in self.basespecs['iris'].iteritems():
            if key not in schema:
                schema[key] = value

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
        as a dictionary for the iris dataset.
        """
        validator = SchemaValidator(specification=self.basespecs['iris'])
        self.assertFalse(validator.is_multifile)
        validated_parser_args = validator.get_parser_args()
        self.assertKwargsEqual(validated_parser_args,
                               self.ideal_iris_parser_args)

    def test_validator_with_specfile_spec(self):
        """Check if the validator works when the specfile and specification are
        both provided.
        """
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
        as a dictionary for the person activity dataset.
        """
        validator = SchemaValidator(
                               specification=self.basespecs['person_activity'])
        self.assertFalse(validator.is_multifile)
        validated = validator.get_parser_args()
        self.assertKwargsEqual(validated, self.ideal_activity_parser_args)

    def test_error_for_relative_filepath(self):
        """Test if validator raises errors when relative paths are found in the
        dictionary.
        """
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
        passed.
        """
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
        provided.
        """
        self.assertRaises(TraitError, SchemaValidator, specfile=self.specfile)

    def test_error_only_name(self):
        """Test if the validator fails when only the path to the specfile is
        provided.
        """
        self.assertRaises(TraitError, SchemaValidator, name="iris")

    def test_validator_specfile_name_iris(self):
        """Test if the validator works when providing specifle and name for the
        iris dataset.
        """
        validator = SchemaValidator(specfile=self.specfile, name="iris")
        validated_parser_args = validator.get_parser_args()
        self.assertKwargsEqual(validated_parser_args,
                               self.ideal_iris_parser_args)

    def test_validator_specfile_name_activity(self):
        """Test if the validator works when providing specifle and name for the
        activity dataset.
        """
        validator = SchemaValidator(specfile=self.specfile,
                                    name="person_activity")
        validated_parser_args = validator.get_parser_args()
        self.assertKwargsEqual(validated_parser_args,
                               self.ideal_activity_parser_args)


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
        """Test if the validator checks for the unique values."""
        validator = SeriesValidator(data=self.species,
                                    rules=self.species_rules)
        cleaned = validator.clean()
        self.assertItemsEqual(cleaned.unique(),
                              self.dataframe['Species'].unique())

    def test_bad_unique_values(self):
        """Test if the validator drops values not specified in the schema."""
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
            series = self.species.unique().tolist()
            validator = SeriesValidator(data=self.species,
                                        rules=self.species_rules)
            cleaned = validator.clean()
            self.assertEqual(cleaned.shape[0], 3)
            self.assertItemsEqual(cleaned.tolist(), series)
        finally:
            self.species_rules['drop_duplicates'] = False

    def test_drop_na(self):
        """Check if the SeriesValidator drops NAs in the series."""
        self.species_rules['drop_na'] = True
        try:
            unqs = np.random.choice(self.species.unique().tolist() + [np.nan],
                                    size=(100,))
            unqs = pd.Series(unqs)
            validator = SeriesValidator(data=unqs,
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
        """Test if the validator enforces min and max values from schema."""
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
        expression provided.
        """
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

    """Tests for the DataFrameValidator class."""

    @classmethod
    def setUpClass(cls):
        cls.maxDiff = None
        with open(TEST_DATA_DICT, 'r') as fileobj:
            basespecs = yaml.load(fileobj, Loader=Loader)
        # Fix the paths in basespecs
        for _, specs in basespecs.iteritems():
            rlpth = specs['path']
            specs['path'] = op.join(op.abspath(op.dirname(__file__)),
                                    rlpth)
        cls._basespecs = basespecs

        iris_validator = SchemaValidator(specification=cls._basespecs['iris'])
        pa_validator = SchemaValidator(
                               specification=cls._basespecs['person_activity'])
        iris_dframe = pd.read_csv(**iris_validator.get_parser_args())
        pa_dframe = pd.read_csv(**pa_validator.get_parser_args())
        cls.iris_dframe = iris_dframe
        cls.pa_dframe = pa_dframe

    def setUp(self):
        self.basespecs = deepcopy(self._basespecs)

    def test_column_rules(self):
        """Test if the DataFrame validator reads and enforces the column rules
        properly.
        """
        dframe_val = DataFrameValidator(data=self.iris_dframe.copy(),
                           column_rules=self.basespecs['iris']['column_rules'])
        cleaned = dframe_val.clean()
        self.assertDataFrameEqual(cleaned, self.iris_dframe.drop_duplicates())
        dframe_val = DataFrameValidator(data=self.pa_dframe.copy(),
                column_rules=self.basespecs['person_activity']['column_rules'])
        cleaned = dframe_val.clean()
        self.assertDataFrameEqual(cleaned, self.pa_dframe.drop_duplicates())

    def test_drop_duplicates(self):
        """Test if the DataFrameValidator is dropping duplicates properly."""
        col_rules = self.basespecs['iris'].get('column_rules')
        data = self.iris_dframe.copy()
        _data = pd.concat((data, data))
        validator = DataFrameValidator(data=_data, column_rules=col_rules)
        cleaned = validator.clean()
        self.assertDataFrameEqual(cleaned, data.drop_duplicates())

    def test_column_exclude_rules(self):
        """Test if the validator drops values excluded from columns."""
        col_rules = deepcopy(self.basespecs['iris']['column_rules'])
        col_rules['Species']['exclude'] = ['virginica', 'versicolor']
        dframe_val = DataFrameValidator(data=self.iris_dframe.copy(),
                                        column_rules=col_rules,
                                        rules={'drop_duplicates': False})
        cleaned_species = dframe_val.clean()['Species']
        self.assertItemsEqual(cleaned_species.unique().tolist(), ['setosa'])
        self.assertEqual(cleaned_species.shape[0], 50)


if __name__ == '__main__':
    unittest.main()
