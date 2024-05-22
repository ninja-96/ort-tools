"""
This module stores addition functions for project.
"""

from typing import Dict
import numpy as np


__all__ = [
    'shape_str_decode'
]


_mapper = {
    'fp16': np.float16,
    'fp32': np.float32,

    'int8': np.int8,
    'int16': np.int16,
    'int32': np.int32,
    'int64': np.int64,

    'uint8': np.uint8,
    'uint16': np.uint16,
    'uint32': np.uint32,
    'uint64': np.uint64,

    'bool': np.bool_
}


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
        name, shape, dtype = input_.split(':')
        input_cfg[name] = {}
        input_cfg[name]['shape'] = []
        input_cfg[name]['dtype'] = None
        for s_item in shape.split('x'):
            input_cfg[name]['shape'].append(int(s_item))
            input_cfg[name]['dtype'] = _mapper[dtype]

    return input_cfg
