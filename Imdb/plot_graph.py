import pickle
import matplotlib.pyplot as plt
import argparse
import os

def get_argument_parser():
    parser = argparse.ArgumentParser();
    parser.add_argument('--input_file', type=str, default='res_markov5.pkl',
                        help='path to input file')
    parser.add_argument('--title', type=str, default='Markov Order 5',
                        help='title for graph')
    return parser

parser = get_argument_parser()
FLAGS = parser.parse_args()
colors = ['b', 'r', 'g', 'k', 'm']

with open(FLAGS.input_file, 'rb') as fp:
    result_dic = pickle.load(fp)

for index, cell in enumerate(['SimpleRNN', 'LSTM', 'GRU']):
	plt.plot(result_dic[cell]['sparse_categorical_accuracy'], label="{} train".format(cell), marker='o', color=colors[index])
	plt.plot(result_dic[cell]['val_sparse_categorical_accuracy'], label="{} val".format(cell), marker='v', linestyle="--", color=colors[index])

plt.ylabel('Accuracy', fontsize=18)
plt.xlabel('Epoch', fontsize=18)
plt.legend(fontsize=12)
plt.grid()
plt.title(FLAGS.title, fontsize=18)
plt.tight_layout()
plt.savefig('{}_Acc.pdf'.format(os.path.splitext(os.path.basename(FLAGS.input_file))[0]))
plt.close()

plt.figure()

for index, cell in enumerate(['SimpleRNN', 'LSTM', 'GRU']):
	plt.plot(result_dic[cell]['loss'], label="{} train".format(cell), marker='o', color=colors[index])
	plt.plot(result_dic[cell]['val_loss'], label="{} val".format(cell), marker='v', linestyle="--", color=colors[index])

plt.ylabel('Loss', fontsize=18)
plt.xlabel('Epoch', fontsize=18)
plt.legend(fontsize=12)
plt.grid()
plt.title(FLAGS.title, fontsize=18)
plt.tight_layout()
plt.savefig('{}_Loss.pdf'.format(os.path.splitext(os.path.basename(FLAGS.input_file))[0]))

plt.close()
