
outext=.txt
npyext=.npy
pklext=.pkl
orders=(3 5 8 10 15 20)

for i in "${orders[@]}"
do
	# python generate_data.py --dtype markov --markovity $i --nsamples 20000 --file_name markov$i$outext
	# python parse_data.py --input_file markov$i$outext --output_file markov$i
	# python learn_seq.py --file_name markov$i$npyext
	python plot_graph.py --input_file res_markov$i$pklext --title "Markov Order $i"
done
