#!/bin/bash

# Experiment: exp1_basic
# Text prompt: photo

python langsam_detector.py -i input -o langsam_experiments/exp1_basic/crops --text-prompt "photo" --visualize --vis-dir langsam_experiments/exp1_basic/visualizations --seed 42 --sample-size 10
