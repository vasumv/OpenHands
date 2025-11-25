"""Dataset for Fray concurrency bug evaluation."""

import json
import os

import pandas as pd


def get_fray_dataset():
    """Load dataset from JSONL file and return as pandas DataFrame for evaluation."""
    dataset_path = os.path.join(os.path.dirname(__file__), 'dataset.jsonl')

    instances = []
    with open(dataset_path, 'r') as f:
        for line in f:
            instances.append(json.loads(line.strip()))

    return pd.DataFrame(instances)
