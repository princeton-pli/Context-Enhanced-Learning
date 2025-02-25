#!/bin/bash
python3 n_layer_dictionary_gen.py  --path=../data/synth_data/ --attr=$1 --nchars=$2 --n_dictionaries=$3 --n_samples_per_dictionary=$4 --depth=$5 --seed=$6 --heldout_ratio=$7