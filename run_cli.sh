#!/bin/bash
# Suppress TensorFlow and Keras logs
export TF_CPP_MIN_LOG_LEVEL=2
export TF_ENABLE_ONEDNN_OPTS=0

# Activate the virtual environment (edit path if needed)
source .venv/bin/activate

# Run the CLI with all passed arguments
python3 -m src.sibyllai_core.cli "$@" 