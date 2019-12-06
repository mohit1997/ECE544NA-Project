import numpy as np
import argparse

def get_argument_parser():
    parser = argparse.ArgumentParser();
    parser.add_argument('--input_file', type=str, default='synthetic.txt',
                        help='path to input file')
    parser.add_argument('--output_file', type=str, default='output',
                        help='path to input file')
    return parser

parser = get_argument_parser()
FLAGS = parser.parse_args()

with open(FLAGS.input_file, 'r', encoding='latin-1') as fp:
    data = fp.read()

vocab_set = list(set(data))
vocab_set.sort()

dic = {vocab_set[i]: i for i in range(len(vocab_set))}

preprocessed_sequence = np.array([dic[i] for i in data])

np.save(FLAGS.output_file, preprocessed_sequence)

