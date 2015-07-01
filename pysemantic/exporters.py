#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2015 jaidev <jaidev@newton>
#
# Distributed under terms of the BSD 3-clause license.

"""
Exporters from PySemantic to databases or other data sinks.
"""


class AbstractExporter(object):
    """Abstract exporter for dataframes that have been cleaned."""

    def get(self, **kwargs):
        raise NotImplementedError

    def set(self, **kwargs):
        raise NotImplementedError


class AerospikeExporter(AbstractExporter):
    """Example class for exporting to an aerospike database."""

    def __init__(self, config, dataframe):
        self.dataframe = dataframe
        self.namespace = config['namespace']
        self.set_name = config['set']
        self.port = config['port']
        self.hostname = config['hostname']

    def set(self, key_tuple, bins):
        self.client.put(key_tuple, bins)

    def run(self):
        import aerospike
        self.client = aerospike.client({'hosts': [(self.hostname,
                                                   self.port)],
                                        'policies':{'timeout': 60000}}).connect()
        for ix in self.dataframe.index:
            self.set((self.namespace, self.set_name, ix),
                     self.dataframe.ix[ix].to_dict())
        self.client.close()
