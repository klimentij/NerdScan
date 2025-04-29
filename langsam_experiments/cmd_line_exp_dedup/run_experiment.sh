#!/bin/bash

# Experiment: cmd_line_exp_dedup
# Text prompt: an old photo.
# Remove overlapping boxes with threshold: 5.0%

python langsam_detector.py -i input -o langsam_experiments/cmd_line_exp_dedup/crops --text-prompt "an old photo." --visualize --vis-dir langsam_experiments/cmd_line_exp_dedup/visualizations --seed 42 --sample-size None --remove-overlaps --overlap-threshold 5.0
