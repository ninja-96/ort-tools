"""
This module stores program that calculate inference speed.
"""

import argparse
import time

import numpy as np
import onnxruntime as ort

from ort_tools.functional import shape_str_decode


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-m', '--model',
        type=str,
        required=True
    )
    parser.add_argument(
        '-w', '--warm-up',
        type=int,
        default=5,
        choices=range(2, 15),
        metavar="[2-15]"
    )
    parser.add_argument(
        '-r', '--repeats',
        type=int,
        default=50,
        choices=range(5, 100),
        metavar="[5-100]"
    )

    parser.add_argument(
        '--shapes',
        type=str,
        required=True
    )

    parser.add_argument(
        '--num-intra-threads',
        type=int,
        default=0
    )
    parser.add_argument(
        '--num-inter-threads',
        type=int,
        default=0
    )
    parser.add_argument(
        '--provider',
        type=str,
        choices=['CPUExecutionProvider', 'CUDAExecutionProvider'],
        default='CPUExecutionProvider'
    )

    args = parser.parse_args()
    print(args)

    sess_options = ort.SessionOptions()

    if args.num_intra_threads > 0:
        sess_options.intra_op_num_threads = args.num_intra_threads
    if args.num_inter_threads > 0:
        sess_options.inter_op_num_threads = args.num_inter_threads

    sess_options.graph_optimization_level = \
        ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    providers = [args.provider]

    model = ort.InferenceSession(
        args.model,
        sess_options=sess_options,
        providers=providers
    )

    input_cfg = shape_str_decode(args.shapes)

    print('Warmup...')
    for _ in range(args.warm_up):
        d = {}
        for n, cfg in input_cfg.items():
            d[n] = np.random.random(size=cfg['shape']).astype(cfg['dtype'])
        r = model.run(None, d)

    print('Benchmark...')
    avg_time = 0
    for _ in range(args.repeats):
        d = {}
        for n, cfg in input_cfg.items():
            d[n] = np.random.random(size=cfg['shape']).astype(cfg['dtype'])

        s = time.time()
        r = model.run(None, d)
        avg_time += time.time() - s

    avg_time /= args.repeats

    d = {}
    for n, s in input_cfg.items():
        d[n] = s

    exec_time = round(avg_time, 5)
    batch_per_second = round(1 / exec_time, 5)

    print(f'Inference time: {exec_time} sec.')
