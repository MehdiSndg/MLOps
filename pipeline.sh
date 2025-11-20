#!/bin/bash
set -e

echo "Running DVC pull..."
dvc pull

echo "Running pipeline..."
python src/train.py

echo "Pipeline completed."
