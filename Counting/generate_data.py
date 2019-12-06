import numpy as np
import argparse
import string
import sys

def get_argument_parser():
	parser = argparse.ArgumentParser();
	parser.add_argument('--dtype', type=str, default='markov',
						help='type of data')
	parser.add_argument('--nsamples', type=int, default=10000,
						help='length of the sequence to be generated')
	parser.add_argument('--markovity', type=int, default=5,
						help='markovity of the generated sequence')
	parser.add_argument('--file_name', type=str, default='synthetic.txt',
						help='name of the output file')
	
	return parser


def markov(markovity, length):
	data = np.empty([length], dtype=np.object_)
	data[:markovity] = np.random.choice(['0', '1'], size=(markovity))
	for i in range(markovity, length):
		if data[i-1] == data[i-markovity]:
			data[i] = '0'
		else:
			data[i] = '1'
	data = "".join((i) for i in data)
	print(len(data))
	return str(data)

def repeat_pattern(pat_len, vocab, length):
	# Keep vocab less than 26
	vals = string.ascii_lowercase[:vocab]
	dic = {i: vals[i] for i in range(vocab)}
	repeats = length // pat_len

	pattern = np.random.choice(vocab, pat_len)

	sequence = np.concatenate([np.tile(pattern, repeats), pattern[:length%pat_len]], axis=0)
	print("Sequence of length {} with vocab {} generated".format(len(sequence), len(np.unique(sequence))))
	sequence = ''.join([dic[i] for i in sequence])

	return sequence

def counting(vocab, length):
	# Keep vocab less than 26
	vals = string.ascii_lowercase[:vocab]
	dic = {i: vals[i] for i in range(vocab)}

	sequence = np.random.randint(vocab, size=length)
	print("Sequence of length {} with vocab {} generated".format(len(sequence), len(np.unique(sequence))))
	sequence = ''.join([dic[i] for i in sequence])

	return sequence



def main():
	np.random.seed(10)
	parser = get_argument_parser()
	FLAGS = parser.parse_args()

	print("Generating sequence....")
	if FLAGS.dtype == "repeat":
		seq = repeat_pattern(10, 3, FLAGS.nsamples)
	elif FLAGS.dtype == "markov":
		seq = markov(FLAGS.markovity, FLAGS.nsamples)
	elif FLAGS.dtype == "count":
		seq = counting(10, FLAGS.nsamples)
	else:
		print("Use a valid dtype, exiting code..")
		sys.exit()
	print("Writing to file {} ....".format(FLAGS.file_name))
	with open(FLAGS.file_name, 'w') as f:
		f.write(seq)
	print("Completed!")




if __name__ == "__main__":
	main()