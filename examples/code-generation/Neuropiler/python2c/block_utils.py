#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function

import ast

LITERAL_NODES = [ast.Num, ast.Str, ast.List, ast.Tuple, ast.Set, ast.Dict,
                 ast.Ellipsis]


def is_literal(node):
    """
    Check if a node represents a python literal.
    """
    return any(map(lambda x: isinstance(node, x), LITERAL_NODES))
