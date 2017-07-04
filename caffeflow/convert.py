#!/usr/bin/env python
from __future__ import absolute_import, division, print_function, unicode_literals

import os
import numpy as np
import argparse
from caffeflow import KaffeError, print_stderr
from caffeflow.tensorflow import TensorFlowTransformer


def fatal_error(msg):
    print_stderr(msg)
    exit(-1)


def validate_arguments(args):
    if (args.data_output_path is not None) and (args.caffemodel is None):
        fatal_error('No input data path provided.')
    if (args.caffemodel is not None) and (args.data_output_path is None):
        fatal_error('No output data path provided.')
    if (args.code_output_path is None) and (args.data_output_path is None):
        fatal_error('No output path specified.')


def convert(def_path, caffemodel_path, data_output_path, code_output_path, phase, use_padding_same=False):
    try:
        transformer = TensorFlowTransformer(def_path, caffemodel_path, phase=phase, use_padding_same=use_padding_same)
        print_stderr('Converting data...')
        if caffemodel_path is not None:
            data = transformer.transform_data()
            print_stderr('Saving data...')
            # create directory if not existing
            dirname = os.path.dirname(data_output_path)
            if not os.path.exists(dirname):
                os.makedirs(dirname)
            # noinspection PyTypeChecker
            with open(data_output_path, 'wb') as data_out:
                np.save(data_out, data)
        if code_output_path:
            print_stderr('Saving source...')
            # create directory if not existing
            dirname = os.path.dirname(code_output_path)
            if not os.path.exists(dirname):
                os.makedirs(dirname)
            # noinspection PyTypeChecker
            with open(code_output_path, 'w') as src_out:
                src_out.write(transformer.transform_source())
        print_stderr('Done.')
    except KaffeError as err:
        fatal_error('Error encountered: {}'.format(err))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('def_path', help='Model definition (.prototxt) path')
    parser.add_argument('--caffemodel', help='Model data (.caffemodel) path')
    parser.add_argument('--data-output-path', help='Converted data output path')
    parser.add_argument('--code-output-path', help='Save generated source to this path')
    parser.add_argument('-p',
                        '--phase',
                        default='test',
                        help='The phase to convert: test (default) or train')
    args = parser.parse_args()
    validate_arguments(args)
    convert(args.def_path, args.caffemodel, args.data_output_path, args.code_output_path, args.phase)


if __name__ == '__main__':
    main()
