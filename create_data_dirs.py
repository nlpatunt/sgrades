#!/usr/bin/env python3
import os

# List of your 12 dataset names
DATASETS = [
    "ASAP-AES",
    "ASAP-SAS",
    "rice_chem",
    "CSEE",
    "EFL",
    "grade_like_a_human_dataset_os",
    "persuade_2",
    "ASAP2",
    "ASAP_plus_plus",
    "SciEntSBank",
    "BEEtlE",
    "automatic_short_answer_grading"
]

# Root folder for data
DATA_ROOT = os.path.join(os.getcwd(), "data")

def make_dirs():
    os.makedirs(DATA_ROOT, exist_ok=True)
    for name in DATASETS:
        path = os.path.join(DATA_ROOT, name)
        os.makedirs(path, exist_ok=True)
        print(f"✔ Created: {path}")

if __name__ == "__main__":
    make_dirs()
