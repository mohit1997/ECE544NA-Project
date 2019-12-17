# ECE544NA-Project


## Requirements
0. GPU (Optional)
1. Python3 (<= 3.6.8)
2. Numpy
3. Argparse
4. Sklearn
5. Tensorflow (gpu) 2


### Download and install dependencies
Download:
```bash
https://github.com/mohit1997/ECE544NA-Project
```

### Run Experiments
1. For counting task
```bash
cd Counting
bash run_experiments.sh
```

2. For learning Markov sequences
```bash
cd Markov
bash run_experiments.sh
```

3. For Cowper Dataset
```bash
cd Cowper
bash run_experiments.sh
```

4. For Imdb Movie Review Classification
```bash
cd Imdb
python learn_seq.py
python plot_graph.py --input_file res_imdb.pkl --title "IMDb Dataset"
```

5. For Reuters Document Classification
```bash
cd Reuters
python learn_seq.py
python plot_graph.py --input_file res_imdb.pkl --title "Reuters Dataset"
```
