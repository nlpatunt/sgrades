from app.config.datasets import get_all_datasets, get_dataset_config

# List all datasets
datasets = get_all_datasets()
for name, config in datasets.items():
    print(f"Dataset: {name}")
    print(f"  Description: {config.description}")
    print(f"  Metrics: {config.evaluation_metrics}")
    print(f"  Score Ranges: {config.score_ranges}")
    print()

# Test specific dataset
asap_config = get_dataset_config("ASAP-AES")
if asap_config:
    print("✅ ASAP-AES loaded successfully!")
    print(f"Description: {asap_config.description}")
else:
    print("❌ ASAP-AES config not found.")