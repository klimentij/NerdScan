#!/bin/bash

# Experiment: exp_dedup
# Text prompt: photo
# Remove overlapping boxes with threshold: 5.0%

python langsam_detector.py -i input -o langsam_experiments/exp_dedup/crops --text-prompt "photo" --visualize --vis-dir langsam_experiments/exp_dedup/visualizations --seed 42 --sample-size 10 --remove-overlaps --overlap-threshold 5.0
