"""
This module stores addition functions for project.
"""

from typing import Dict


def shape_str_decode(shapes: str) -> Dict:
    """
    shape_str_decode(shapes: str) -> Dict.

    Parse input string with input name and shape to Dict.

    Args:
        shapes(str): input string to parse.

    Return:
        Dict: input name (key) and their shape (value).
    """
    shapes = shapes.replace(' ', '')

    input_cfg = {}
    for input_ in shapes.split(','):
        name, shape = input_.split(':')
        input_cfg[name] = []
        for s_item in shape.split('x'):
            input_cfg[name].append(int(s_item))

    return input_cfg
