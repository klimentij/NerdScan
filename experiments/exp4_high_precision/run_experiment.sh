#!/bin/bash

# Experiment: exp4_high_precision
# Text prompt: a photo. a picture. a photograph.
# Box threshold: 0.4
# Text threshold: 0.3

python groundedsam_detector.py -i input -o experiments/exp4_high_precision/crops --text-prompt "a photo. a picture. a photograph." --box-threshold 0.4 --text-threshold 0.3 --visualize --vis-dir experiments/exp4_high_precision/visualizations --seed 42 --sample-size 10
