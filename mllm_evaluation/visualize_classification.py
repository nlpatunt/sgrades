import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# =========================
#   ASAG CLASSIFICATION DATASETS (F1 SCORES)
# =========================

classification_datasets = [
    "BEEtlE", "SciEntSBank"
]

models = ["GPT-4o-mini", "Gemini-2.5-Flash", "LLaMA-4-Scout"]
strategies = ["Ind", "Ded", "Abd", "Ind+Abd", "Ind+Ded", "Ded+Abd"]

# =========================
#   CLASSIFICATION DATA (F1)
# =========================

class_pairs = [(d, m) for d in classification_datasets for m in models]

classification_data = {
    "Dataset": [p[0] for p in class_pairs],
    "Model": [p[1] for p in class_pairs],
    
    "Ind": [
        # BEEtlE
        0.5602, 0.6408, 0.5406,
        # SciEntSBank
        0.6495, 0.7279, 0.6573,
    ],
    
    "Ded": [
        # BEEtlE_Avg
        0.5217, 0.6167, 0.5396,
        # SciEntSBank_Avg
        0.6633, 0.7415, 0.6603,
    ],
    
    "Abd": [
        # BEEtlE_Avg
        0.5328, 0.5919, 0.3233,
        # SciEntSBank_Avg
        0.6488, 0.7169, 0.6631,
    ],
    
    "Ind+Abd": [
        # BEEtlE_Avg
        0.5280, 0.6167, 0.4679,
        # SciEntSBank_Avg
        0.6504, 0.7224, 0.6395,
    ],
    
    "Ind+Ded": [
        # BEEtlE_Avg
        0.5430, 0.6371, 0.4830,
        # SciEntSBank_Avg
        0.6530, 0.7210, 0.6473,
    ],
    
    "Ded+Abd": [
        # BEEtlE_Avg
        0.5262, 0.5004, 0.3574,
        # SciEntSBank_Avg
        0.6441, 0.6897, 0.6202,
    ],
}

df_classification = pd.DataFrame(classification_data)

# =========================
#   CREATE HEATMAPS
# =========================

sns.set_theme(style="whitegrid")
cmap = sns.color_palette("YlGnBu", as_cmap=True)

fig, axes = plt.subplots(2, 1, figsize=(8, 8), sharex=True, sharey=True)

for ax, dataset in zip(axes, classification_datasets):
    subset = df_classification[df_classification["Dataset"] == dataset].set_index("Model")[strategies]
    
    sns.heatmap(
        subset, annot=True, cmap=cmap, fmt=".2f",
        cbar=False, ax=ax, linewidths=0.5, linecolor="white",
        annot_kws={"size": 11, "weight": "bold"},
        vmin=0.3, vmax=0.8,
        square=True
    )
    
    # Color text based on value
    for text in ax.texts:
        try:
            val = float(text.get_text())
            text.set_color('white' if val > 0.55 else 'black')
        except:
            pass
    
    ax.set_title(dataset, fontsize=13, weight="bold")
    ax.set_ylabel("")
    ax.tick_params(axis="x", rotation=40, labelsize=10)
    ax.tick_params(axis="y", labelsize=10)

# =========================
#   SHARED COLORBAR
# =========================

cbar_ax = fig.add_axes([0.92, 0.25, 0.03, 0.5])
norm = plt.Normalize(vmin=0.3, vmax=0.8)
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
cbar = plt.colorbar(sm, cax=cbar_ax)
cbar.set_label("F1 Score", fontsize=11, weight="bold")

# =========================
#   MAIN TITLE
# =========================

fig.suptitle(
    "ASAG Classification — F1 Score Performance",
    fontsize=16, weight="bold", y=0.98
)

plt.subplots_adjust(left=0.08, right=0.89, top=0.88, bottom=0.15, wspace=0.25)

plt.savefig("/home/ts1506.UNT/Desktop/Work/classification_heatmaps_final.pdf", dpi=400, bbox_inches='tight')
plt.show()