#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2015 jaidev <jaidev@newton>
#
# Distributed under terms of the MIT license.

"""
Misecellaneous bells and whistles
"""

import json


class TypeEncoder(json.JSONEncoder):

    def default(self, obj):
        if isinstance(obj, type):
            return str(obj)
        else:
            return json.JSONEncoder.default(self, obj)
