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
import sys

# Deterministic Results
tf.random.set_seed(0)


def get_argument_parser():
	parser = argparse.ArgumentParser();
	parser.add_argument('--epochs', type=int, default=10,
						help='Number of epochs')
	
	return parser


def SeqLearner(vocab_size, timesteps, num_classes, cell_type='LSTM'):
	# inp = layers.Input(shape=(timesteps,))
	if stateful:
		inp = layers.Input(batch_shape=(batch_size, timesteps,))
	else:
		inp = layers.Input(shape=(None,))
	# batch_shape
	x = layers.Embedding(vocab_size, 16)(inp)
	# output, state_h, state_c = layers.LSTM(
	# 	64, return_state=True, name='encoder')(x)
	x = getattr(layers, cell_type)(
		32, stateful=stateful, return_sequences=True)(x)
	x = getattr(layers, cell_type)(
		32, stateful=stateful, return_sequences=False)(x)

	output = layers.Dense(num_classes, activation='softmax')(x)

	return Model(inp, output)



batch_size = 64
stateful = False
timesteps = 200

parser = get_argument_parser()
FLAGS = parser.parse_args()

(X_train, y_train), (X_test, y_test) = tf.keras.datasets.reuters.load_data(
	path='reuters.npz',
	num_words=None,
	skip_top=0,
	maxlen=timesteps,
	test_split=0.2,
	seed=113,
	start_char=1,
	oov_char=2,
	index_from=3,
)




X_train = tf.keras.preprocessing.sequence.pad_sequences(X_train, value=0, maxlen=None)
X_test = tf.keras.preprocessing.sequence.pad_sequences(X_test, value=0, maxlen=X_train.shape[1])

vocab_size = np.max(X_train) + 1
num_classes = np.max(y_train) + 1

print(vocab_size, num_classes)
print(X_train.shape)


result_dic = {}

for cell in ['SimpleRNN', 'LSTM', 'GRU']:
	model = SeqLearner(vocab_size, timesteps, num_classes, cell_type=cell)
	model.compile(optimizer=keras.optimizers.RMSprop(learning_rate=1e-3),
				  loss=keras.losses.SparseCategoricalCrossentropy(),
				  metrics=[keras.metrics.SparseCategoricalAccuracy()])

	if stateful:
		l = len(X_train) // batch_size * batch_size
		X_train = X_train[:l]
		y_train = y_train[:l]

		l = len(X_test) // batch_size * batch_size

		X_test = X_test[:l]
		y_test = y_test[:l]

	history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=FLAGS.epochs, batch_size=batch_size)
	result_dic[cell] = history.history


with open('res_{}.pkl'.format('reuters'), 'wb') as fp:
	pickle.dump(result_dic, fp, protocol=pickle.HIGHEST_PROTOCOL)








