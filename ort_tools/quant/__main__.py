"""
This module stores program that quantize model.
"""

import argparse
import os

from onnxruntime.quantization import quantize_static, QuantType, QuantFormat

from ort_tools.functional import shape_str_decode
from ort_tools.quant.reader import RandomDataDataReader


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-m', '--model',
        type=str,
        required=True
    )
    parser.add_argument(
        '--shapes',
        type=str,
        required=True
    )

    parser.add_argument(
        '--count',
        type=int,
        default=20
    )
    parser.add_argument(
        '--not-per-channel',
        action='store_true'
    )
    parser.add_argument(
        '--quant-format',
        type=str,
        choices=['QOperator', 'QDQ'],
        default='QOperator'
    )
    parser.add_argument(
        '--activation-type',
        type=str,
        choices=['QUInt8', 'QInt8'],
        default='QUInt8'
    )
    parser.add_argument(
        '--weight-type',
        type=str,
        choices=['QUInt8', 'QInt8'],
        default='QUInt8'
    )

    args = parser.parse_args()
    print(args)

    input_cfg = shape_str_decode(args.shapes)
    reader = RandomDataDataReader(input_cfg, args.count)

    file_path = os.path.dirname(args.model)
    file_basename = '.'.join(os.path.basename(args.model).split('.')[:-1])
    quant_file_path = os.path.normpath(
        f'{file_path}/{file_basename}_quant.onnx'
    )

    quant_format_map = {
        'QOperator': QuantFormat.QOperator,
        'QDQ': QuantFormat.QDQ
    }

    quant_type_map = {
        'QUInt8': QuantType.QUInt8,
        'QInt8': QuantType.QInt8
    }

    quantize_static(
        args.model,
        quant_file_path,
        reader,
        quant_format=quant_format_map[args.quant_format],
        per_channel=not args.not_per_channel,
        activation_type=quant_type_map[args.activation_type],
        weight_type=quant_type_map[args.weight_type]
    )
