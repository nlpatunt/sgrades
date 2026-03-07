"""
Pre-download all datasets from HuggingFace and save as CSV files locally.
Run this once to populate the cache, then the download endpoint will serve from disk.
"""
import os
import json
import pandas as pd
from pathlib import Path
from datasets import load_dataset

CACHE_DIR = "/home/ts1506.unt.ad.unt.edu/sgrades/dataset_cache"
HF_TOKEN = os.getenv("HUGGINGFACE_TOKEN")

# Dataset configurations
DATASETS = [
    {"name": "D_ASAP-AES", "hf_id": "nlpatunt/D_ASAP-AES", "config": None},
    {"name": "D_ASAP2", "hf_id": "nlpatunt/D_ASAP2", "config": None},
    {"name": "D_ASAP_plus_plus", "hf_id": "nlpatunt/D_ASAP_plus_plus", "config": None},
    {"name": "D_BEEtlE_2way", "hf_id": "nlpatunt/D_BEEtlE", "config": "2way"},
    {"name": "D_BEEtlE_3way", "hf_id": "nlpatunt/D_BEEtlE", "config": "3way"},
    {"name": "D_CSEE", "hf_id": "nlpatunt/D_CSEE", "config": None},
    {"name": "D_ASAP-SAS", "hf_id": "nlpatunt/D_ASAP-SAS", "config": None},
    {"name": "D_Mohlar", "hf_id": "nlpatunt/D_Mohlar", "config": None},
    {"name": "D_Ielts_Writing_Dataset", "hf_id": "nlpatunt/D_Ielts_Writing_Dataset", "config": None},
    {"name": "D_OS_Dataset_q1", "hf_id": "nlpatunt/D_OS_Dataset", "config": "q1"},
    {"name": "D_OS_Dataset_q2", "hf_id": "nlpatunt/D_OS_Dataset", "config": "q2"},
    {"name": "D_OS_Dataset_q3", "hf_id": "nlpatunt/D_OS_Dataset", "config": "q3"},
    {"name": "D_OS_Dataset_q4", "hf_id": "nlpatunt/D_OS_Dataset", "config": "q4"},
    {"name": "D_OS_Dataset_q5", "hf_id": "nlpatunt/D_OS_Dataset", "config": "q5"},
    {"name": "D_Ielts_Writing_Task_2_Dataset", "hf_id": "nlpatunt/D_Ielts_Writing_Task_2_Dataset", "config": None},
    {"name": "D_persuade_2", "hf_id": "nlpatunt/D_persuade_2", "config": None},
    {"name": "D_SciEntSBank_2way", "hf_id": "nlpatunt/D_SciEntSBank", "config": "2way"},
    {"name": "D_SciEntSBank_3way", "hf_id": "nlpatunt/D_SciEntSBank", "config": "3way"},
    {"name": "D_Regrading_Dataset_J2C", "hf_id": "nlpatunt/D_Regrading_Dataset_J2C", "config": None},
    {"name": "D_Rice_Chem_Q1", "hf_id": "nlpatunt/D_Rice_Chem", "config": "Q1"},
    {"name": "D_Rice_Chem_Q2", "hf_id": "nlpatunt/D_Rice_Chem", "config": "Q2"},
    {"name": "D_Rice_Chem_Q3", "hf_id": "nlpatunt/D_Rice_Chem", "config": "Q3"},
    {"name": "D_Rice_Chem_Q4", "hf_id": "nlpatunt/D_Rice_Chem", "config": "Q4"},
]

def cache_dataset(dataset_info):
    name = dataset_info["name"]
    hf_id = dataset_info["hf_id"]
    config = dataset_info["config"]
    
    dataset_dir = Path(CACHE_DIR) / name
    dataset_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if already cached
    if (dataset_dir / "train.csv").exists():
        print(f"✅ Already cached: {name}")
        return
    
    print(f"📥 Downloading: {name}...")
    
    try:
        splits = ["train", "validation", "test"]
        for split in splits:
            try:
                if config:
                    ds = load_dataset(hf_id, config, split=split, token=HF_TOKEN)
                else:
                    ds = load_dataset(hf_id, split=split, token=HF_TOKEN)
                
                df = ds.to_pandas()
                
                # Clear scores for test split
                if split == "test":
                    score_cols = [c for c in df.columns if "score" in c.lower() or "label" in c.lower() or "grade" in c.lower()]
                    for col in score_cols:
                        df[col] = ""
                
                df.to_csv(dataset_dir / f"{split}.csv", index=False)
                print(f"  ✅ Saved {split}: {len(df)} rows")
            except Exception as e:
                print(f"  ⚠️ No {split} split: {e}")
    except Exception as e:
        print(f"  ❌ Failed: {e}")

if __name__ == "__main__":
    Path(CACHE_DIR).mkdir(parents=True, exist_ok=True)
    print(f"📁 Cache directory: {CACHE_DIR}")
    
    for i, dataset in enumerate(DATASETS):
        print(f"\n[{i+1}/{len(DATASETS)}] Processing {dataset['name']}...")
        cache_dataset(dataset)
    
    print("\n🎉 Done! All datasets cached.")
