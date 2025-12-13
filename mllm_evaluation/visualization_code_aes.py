import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# =========================
#   AES DATASETS (6 datasets × 3 models × 6 strategies)
# =========================

datasets = [
    "ASAP-AES", "ASAP2", "ASAP++",
    "Persuade-2", "IELTS_Writing_Dataset", "IELTS_Writing_Task_2_Dataset"
]

models = ["GPT-4o-mini", "Gemini-2.5-Flash", "LLaMA-4-Scout"]

# Create dataset-model pairs
pairs = [(d, m) for d in datasets for m in models]

# QWK scores for all combinations - VERIFIED FROM SPREADSHEETS
data = {
    "Dataset": [p[0] for p in pairs],
    "Model": [p[1] for p in pairs],
    
    # Inductive
    "Ind": [
        # ASAP-AES
        0.9207, 0.6046, 0.6209,
        # ASAP2
        0.2889, 0.3338, 0.4128,
        # ASAP++
        0.1914, 0.3033, 0.2423,
        # Persuade-2
        0.6727, 0.7754, 0.2962,
        # IELTS_Writing_Dataset
        0.4418, 0.4855, 0.3662,
        # IELTS_Writing_Task_2_Dataset
        0.1901, 0.2030, 0.1790
    ],
    
    # Deductive
    "Ded": [
        # ASAP-AES
        0.9523, 0.8835, 0.8620,
        # ASAP2
        0.1650, 0.2389, 0.1456,
        # ASAP++
        0.1856, 0.1704, 0.2233,
        # Persuade-2
        0.5304, 0.7003, 0.4688,
        # IELTS_Writing_Dataset
        0.2857, 0.3691, 0.5455,
        # IELTS_Writing_Task_2_Dataset
        0.2340, 0.1876, 0.2202
    ],
    
    # Abductive
    "Abd": [
        # ASAP-AES
        0.9542, 0.8751, 0.9112,
        # ASAP2
        0.1595, 0.2009, 0.1761,
        # ASAP++
        0.1914, 0.1407, 0.2198,
        # Persuade-2
        0.4856, 0.6805, 0.3814,
        # IELTS_Writing_Dataset
        0.2820, 0.4346, 0.3144,
        # IELTS_Writing_Task_2_Dataset
        0.2374, 0.1659, 0.2219
    ],
    
    # Ind+Abd
    "Ind+Abd": [
        # ASAP-AES
        0.9151, 0.7177, 0.8863,
        # ASAP2
        0.2827, 0.3806, 0.3355,
        # ASAP++
        0.1931, 0.2864, 0.2321,
        # Persuade-2
        0.6667, 0.7721, 0.2606,
        # IELTS_Writing_Dataset
        0.3529, 0.4737, 0.273,
        # IELTS_Writing_Task_2_Dataset
        0.1856, 0.1983, 0.1162
    ],
    
    # Ind+Ded
    "Ind+Ded": [
        # ASAP-AES
        0.95, 0.8182, 0.9371,
        # ASAP2
        0.25, 0.3455, 0.1937,
        # ASAP++
        0.21, 0.2303, 0.2292,
        # Persuade-2 - NEED TO VERIFY THIS VALUE
        0.67, 0.7671, 0.3181,  # GPT-4o-mini value needs verification
        # IELTS_Writing_Dataset
        0.41, 0.4419, 0.6429,
        # IELTS_Writing_Task_2_Dataset
        0.18, 0.1903, 0.2584
    ],
    
    # Ded+Abd
    "Ded+Abd": [
        # ASAP-AES
        0.9159, 0.8112, 0.9189,
        # ASAP2
        0.1051, 0.1824, 0.1557,
        # ASAP++
        0.1514, 0.1379, 0.2390,
        # Persuade-2
        0.5410, 0.5875, 0.3733,
        # IELTS_Writing_Dataset
        0.2667, 0.2168, 0.4961,
        # IELTS_Writing_Task_2_Dataset
        0.1940, 0.1394, 0.2519
    ],
}

df = pd.DataFrame(data)

strategies = ["Ind", "Ded", "Abd", "Ind+Abd", "Ind+Ded", "Ded+Abd"]

# =========================
#   Create Heatmaps
# =========================
sns.set_theme(style="whitegrid")
cmap = sns.color_palette("YlGnBu", as_cmap=True)
norm = plt.Normalize(vmin=0.1, vmax=1.0)

fig, axes = plt.subplots(3, 2, figsize=(12, 10), sharex=True, sharey=True)
axes = axes.flatten()

for ax, dataset in zip(axes, datasets):
    subset = df[df["Dataset"] == dataset].set_index("Model")[strategies]
    
    sns.heatmap(
        subset, annot=True, cmap=cmap, fmt=".2f",
        cbar=False, ax=ax, linewidths=0.5, linecolor="white",
        annot_kws={"size": 11, "weight": "bold"},
        vmin=0.1, vmax=1.0,
        square=True
    )

    # Color text based on value
    for text in ax.texts:
        val = float(text.get_text())
        text.set_color('white' if val > 0.5 else 'black')

    ax.set_title(dataset, fontsize=13, weight="bold")
    ax.set_ylabel("")
    ax.tick_params(axis="x", rotation=40, labelsize=10)
    ax.tick_params(axis="y", labelsize=10)

# =========================
#   Shared colorbar
# =========================
cbar_ax = fig.add_axes([0.93, 0.25, 0.02, 0.5])
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
cbar = plt.colorbar(sm, cax=cbar_ax)
cbar.set_label("Quadratic Weighted Kappa (QWK)", fontsize=11, weight="bold")

# =========================
#   Title
# =========================
fig.suptitle(
    "AES — Model-wise QWK Across Reasoning Strategies per Dataset",
    fontsize=16, weight="bold", y=0.98
)

plt.subplots_adjust(left=0.08, right=0.91, top=0.94, bottom=0.08, hspace=0.35, wspace=0.25)

plt.savefig("/home/ts1506.UNT/Desktop/Work/AES_modelwise_heatmaps_final.pdf", dpi=400, bbox_inches='tight')
plt.show()