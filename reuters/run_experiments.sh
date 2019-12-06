outext=.txt
npyext=.npy
pklext=.pkl
# orders=(5 10 15)

# for i in "${orders[@]}"
# do
# 	python generate_data.py --dtype markov --markovity $i --nsamples 20000 --file_name markov$i$outext
# 	python parse_data.py --input_file markov$i$outext --output_file markov$i
# 	python learn_seq.py --file_name markov$i$npyext
# 	python plot_graph.py --input_file res_markov$i$pklext --title "Markov Order $i"
# done

# # Download link
# wget https://storage.googleapis.com/download.tensorflow.org/data/illiad/cowper.txt
orders=(cowper)

for i in "${orders[@]}"
do
	python parse_data.py --input_file $i$outext --output_file $i
	python learn_seq.py --file_name $i$npyext --epochs 5
	python plot_graph.py --input_file res_$i$pklext --title "$i"
done
