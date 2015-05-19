#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2015 jaidev <jaidev@schrodinger.local>
#
# Distributed under terms of the MIT license.


from pysemantic.project import Project

__version__ = "0.1.0"


def test():
    """Interactive loader for tests."""
    import unittest
    unittest.main(module='pysemantic.tests', exit=False)

__all__ = ['Project', 'test']
