#!/bin/sh

# Replace environment variables in the Python command
python server.py --hidden_neurons ${HIDDEN_NEURONS} --limit_per_digit ${LIMIT_PER_DIGIT} --num_classes ${NUM_CLASSES}
