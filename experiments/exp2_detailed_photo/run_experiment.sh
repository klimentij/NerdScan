#!/bin/bash

# Experiment: exp2_detailed_photo
# Text prompt: a photo. a picture. a photograph. a snapshot. an image. a portrait.
# Box threshold: 0.3
# Text threshold: 0.2

python groundedsam_detector.py -i input -o experiments/exp2_detailed_photo/crops --text-prompt "a photo. a picture. a photograph. a snapshot. an image. a portrait." --box-threshold 0.3 --text-threshold 0.2 --visualize --vis-dir experiments/exp2_detailed_photo/visualizations --seed 42 --sample-size 10
