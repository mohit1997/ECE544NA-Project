import numpy as np
import argparse
from utils import *
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import Model
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
import pickle
import os

# Deterministic Results
tf.random.set_seed(0)


def get_argument_parser():
	parser = argparse.ArgumentParser();
	parser.add_argument('--file_name', type=str, default='markov10.npy',
						help='name of the output file')
	parser.add_argument('--ts', type=int, default=16,
						help='number of timesteps to count')
	parser.add_argument('--epochs', type=int, default=10,
						help='number of epochs to train')
	
	return parser

def get_counts(X):
	nums = np.sort(np.unique(X))
	counts = [np.count_nonzero(X==i, axis=1) for i in nums]
	counts = np.vstack(counts).T
	return counts



def SeqLearner(vocab_size, timesteps, cell_type='LSTM'):
	# inp = layers.Input(shape=(timesteps,))
	if stateful:
		inp = layers.Input(batch_shape=(batch_size, timesteps,))
	else:
		inp = layers.Input(shape=(timesteps,))
	# batch_shape
	x = layers.Embedding(vocab_size, 16)(inp)
	# output, state_h, state_c = layers.LSTM(
	# 	64, return_state=True, name='encoder')(x)
	x = getattr(layers, cell_type)(
		32, stateful=stateful, return_sequences=True)(x)
	x = getattr(layers, cell_type)(
		32, stateful=stateful, return_sequences=False)(x)

	output = layers.Dense(vocab_size)(x)

	return Model(inp, output)



parser = get_argument_parser()
FLAGS = parser.parse_args()

batch_size = 64
timesteps = FLAGS.ts
stateful = False

seq = np.load(FLAGS.file_name)
vocab_size = len(np.unique(seq))

X,_ = generate_single_output_data(seq, batch_size, timesteps)
Y = get_counts(X)


X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, shuffle=False, random_state=42)

result_dic = {}

for cell in ['SimpleRNN', 'LSTM', 'GRU']:
	model = SeqLearner(vocab_size, timesteps, cell_type=cell)
	model.compile(optimizer=keras.optimizers.RMSprop(learning_rate=1e-3),
				  loss=keras.losses.MSE
				  )

	if stateful:
		l = len(X_train) // batch_size * batch_size
		X_train = X_train[:l]
		y_train = y_train[:l]

		l = len(X_test) // batch_size * batch_size

		X_test = X_test[:l]
		y_test = y_test[:l]

	history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=FLAGS.epochs, batch_size=batch_size)
	result_dic[cell] = history.history


with open('res_{}{}.pkl'.format(os.path.splitext(os.path.basename(FLAGS.file_name))[0], timesteps), 'wb') as fp:
    pickle.dump(result_dic, fp, protocol=pickle.HIGHEST_PROTOCOL)








