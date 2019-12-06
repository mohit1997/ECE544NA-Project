
outext=.txt
npyext=.npy
pklext=.pkl
orders=(10 20 30 40)

for i in "${orders[@]}"
do
	python generate_data.py --dtype count --nsamples 20000 --file_name counts$outext
	python parse_data.py --input_file counts$outext --output_file counts
	python learn_seq.py --file_name counts$npyext --ts $i --epochs 20
	python plot_graph.py --input_file res_counts$i$pklext --title "Counts for length $i"
done
