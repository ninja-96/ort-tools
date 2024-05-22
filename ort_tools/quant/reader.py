"""
This module stores random data reader (generator) for quantization module.
"""


from typing import Dict

import numpy as np
from onnxruntime.quantization import CalibrationDataReader


class RandomDataDataReader(CalibrationDataReader):
    """
    RandomDataDataReader(CalibrationDataReader).

    This class implements data reader for ONNXRuntime quantization module.
    `RandomDataDataReader` generates random tensor and return it.
    """
    def __init__(self, input_cfg: Dict, count: int = 100):
        self.enum_data = None
        self.data = []
        for _ in range(count):
            tmp = {}
            for n, cfg in input_cfg.items():
                tmp[n] = np.random.random(size=cfg['shape']).astype(cfg['dtype'])
            self.data.append(tmp)

    def get_next(self):
        if self.enum_data is None:
            self.enum_data = iter([d for d in self.data])
        return next(self.enum_data, None)

    def rewind(self):
        self.enum_data = None
