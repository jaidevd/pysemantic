#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2015 jaidev <jaidev@newton>
#
# Distributed under terms of the BSD 3-clause license.

"""
Misecellaneous bells and whistles.
"""

import json
import pandas as pd
import numpy as np
import datetime

DATA_TYPES = {'String': str, 'Date/Time': datetime.date, 'Float': float,
              'Integer': int}


class TypeEncoder(json.JSONEncoder):

    def default(self, obj):
        if isinstance(obj, type):
            return str(obj)
        elif isinstance(obj, set):
            return list(obj)
        elif callable(obj):
            return ".".join((obj.__module__, obj.__name__))
        elif isinstance(obj, np.ndarray):
            return np.array_str(obj)
        else:
            return json.JSONEncoder.default(self, obj)


def generate_questionnaire(filepath):
    """Generate a questionnaire for data at `filepath`.

    This questionnaire will be presented to the client, which helps us
    automatically generate the schema.

    :param filepath: Path to the file that needs to be ingested.
    :type filepath: str
    :return: A dictionary of questions and their possible answers. The format
    of the dictionary is such that every key is a question to be put to the
    client, and its value is a list of possible answers. The first item in the
    list is the default value.
    :rtype: dict
    """
    qdict = {}
    if filepath.endswith(".tsv"):
        dataframe = pd.read_table(filepath)
    else:
        dataframe = pd.read_csv(filepath)
    for col in dataframe.columns:
        qstring = "What is the data type of {}?".format(col)
        if "float" in str(dataframe[col].dtype).lower():
            defaultType = "Float"
        elif "object" in str(dataframe[col].dtype).lower():
            defaultType = "String"
        elif "int" in str(dataframe[col].dtype).lower():
            defaultType = "Integer"
        typeslist = DATA_TYPES.keys()
        typeslist.remove(defaultType)
        typeslist = [defaultType] + typeslist
        qdict[qstring] = typeslist
    return qdict


def colnames(filename, parser=None, **kwargs):
    """
    Read the column names of a delimited file, without actually reading the
    whole file. This is simply a wrapper around `pandas.read_csv`, which reads
    only one row and returns the column names.


    :param filename: Path to the file to be read
    :param kwargs: Arguments to be passed to the `pandas.read_csv`
    :type filename: str
    :rtype: list

    :Example:

    Suppose we want to see the column names of the Fisher iris dataset.

    >>> colnames("/path/to/iris.csv")
    ['Sepal Length', 'Petal Length', 'Sepal Width', 'Petal Width', 'Species']

    """

    if 'nrows' in kwargs:
        UserWarning("The nrows parameter is pointless here. This function only"
                    "reads one row.")
        kwargs.pop('nrows')
    if parser is None:
        if "sep" in kwargs:
            sep = kwargs.get('sep')
            if sep == r"\t":
                parser = pd.read_table
            else:
                parser = pd.read_csv
        elif filename.endswith('.tsv'):
            parser = pd.read_table
        else:
            parser = pd.read_csv
    return parser(filename, nrows=1, **kwargs).columns.tolist()


def get_md5_checksum(filepath):
    """Get the md5 checksum of a file.

    :param filepath: Path to the file of which to calculate the md5 checksum.
    :type filepath: Str
    :return: MD5 checksum of the file.
    :rtype: Str
    :Example:

    >>> get_md5_checksum('pysemantic/tests/testdata/iris.csv')
    '9b3ecf3031979169c0ecc5e03cfe20a6'

    """
    import subprocess
    cmd = "md5sum {}".format(filepath).split()
    return subprocess.check_output(cmd).rstrip().split()[0]
