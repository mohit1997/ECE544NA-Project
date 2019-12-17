outext=.txt
npyext=.npy
pklext=.pkl


# Download link
wget https://storage.googleapis.com/download.tensorflow.org/data/illiad/cowper.txt
orders=(cowper)

for i in "${orders[@]}"
do
	python parse_data.py --input_file $i$outext --output_file $i
	python learn_seq.py --file_name $i$npyext --epochs 10
	python plot_graph.py --input_file res_$i$pklext --title "Cowper Dataset"
done
