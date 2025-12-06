from datasets import load_dataset
import pandas as pd
import os
from huggingface_hub import login

# -----------------------------------------------------------
# 1. Authenticate (required for private Hugging Face datasets)
# -----------------------------------------------------------
# Replace with your actual token or leave blank to be prompted
login(token="REMOVED_KEY")

# -----------------------------------------------------------
# 2. Define dataset list
# -----------------------------------------------------------
dataset_names = [
    # "nlpatunt/D_ASAP-AES",
    # "nlpatunt/D_ASAP2",
    # "nlpatunt/D_ASAP_plus_plus",
    # "nlpatunt/D_persuade_2",
    # "nlpatunt/D_Ielts_Writing_Dataset",
    # "nlpatunt/D_Ielts_Writing_Task_2_Dataset",
    # "nlpatunt/D_Regrading_Dataset_J2C",
    # "nlpatunt/D_ASAP-SAS",
    # "nlpatunt/D_CSEE",
    # "nlpatunt/D_Mohlar"
]

SAVE_DIR = "sampleee_datasets"
os.makedirs(SAVE_DIR, exist_ok=True)

# -----------------------------------------------------------
# 3. Helper function: load test split and take 21 %
# -----------------------------------------------------------
def sample_and_save(dataset_id, subset_name=None):
    try:
        # for multi-config datasets like Rice_Chem/Q1 etc.
        ds = load_dataset(dataset_id, name=subset_name, split="test", trust_remote_code=True)
    except Exception:
        # fallback for single-config datasets
        ds = load_dataset(dataset_id, split="test", trust_remote_code=True)

    df = ds.to_pandas()
    sample_df = df.sample(frac=0.21, random_state=42)

    fname = dataset_id.split("/")[-1]
    if subset_name:
        fname = f"{fname}_{subset_name}"

    out_path = os.path.join(SAVE_DIR, f"{fname}.csv")
    sample_df.to_csv(out_path, index=False)
    print(f"✅ Saved {len(sample_df)} rows → {out_path}")

# -----------------------------------------------------------
# 4. Process all regular datasets
# -----------------------------------------------------------
for d in dataset_names:
    sample_and_save(d)

for config in ["2way", "3way"]:
    #sample_and_save("nlpatunt/D_BEEtlE", subset_name=config)
    sample_and_save("nlpatunt/D_SciEntSBank", subset_name=config)

# -----------------------------------------------------------
# 5. Process OS_Dataset (Q1–Q5)
# -----------------------------------------------------------
# for q in ["q1", "q2", "q3", "q4", "q5"]:
#     sample_and_save("nlpatunt/D_OS_Dataset", subset_name=q)

# -----------------------------------------------------------
# 6. Process Rice_Chem (Q1–Q4)
# -----------------------------------------------------------
# for q in ["Q1", "Q2", "Q3", "Q4"]:
#     sample_and_save("nlpatunt/D_Rice_Chem", subset_name=q)

print("\n🎉 Done! All 21% test samples saved in the 'sampled_datasets/' folder.")
