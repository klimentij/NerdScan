#!/bin/bash

# Experiment: exp1_basic_photo
# Text prompt: a photo. a picture. a photograph.
# Box threshold: 0.35
# Text threshold: 0.25

python groundedsam_detector.py -i input -o experiments/exp1_basic_photo/crops --text-prompt "a photo. a picture. a photograph." --box-threshold 0.35 --text-threshold 0.25 --visualize --vis-dir experiments/exp1_basic_photo/visualizations --seed 42 --sample-size 10
