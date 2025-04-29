#!/bin/bash

# Experiment: exp3_vintage_photo
# Text prompt: an old photo. a vintage photograph. a historical picture. a family photo.
# Box threshold: 0.3
# Text threshold: 0.2

python groundedsam_detector.py -i input -o experiments/exp3_vintage_photo/crops --text-prompt "an old photo. a vintage photograph. a historical picture. a family photo." --box-threshold 0.3 --text-threshold 0.2 --visualize --vis-dir experiments/exp3_vintage_photo/visualizations --seed 42 --sample-size 10
