#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2015 jaidev <jaidev@newton>
#
# Distributed under terms of the MIT license.

"""
Misecellaneous bells and whistles.
"""

import json


class TypeEncoder(json.JSONEncoder):

    def default(self, obj):
        if isinstance(obj, type):
            return str(obj)
        elif isinstance(obj, set):
            return list(obj)
        elif callable(obj):
            return ".".join((obj.__module__, obj.__name__))
        else:
            return json.JSONEncoder.default(self, obj)


def colnames(filename, **kwargs):
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
    import pandas as pd
    return pd.read_csv(filename, nrows=1, **kwargs).columns.tolist()


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
    import hashlib
    with open(filepath, "rb") as fid:
        checksum = hashlib.md5(fid.read()).hexdigest()
    return checksum
