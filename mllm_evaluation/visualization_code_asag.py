import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# =========================
#   ASAG REGRESSION DATASETS ONLY
# =========================

regression_datasets = [
    "ASAP-SAS", "CSEE", "Mohlar", "Regrading_Dataset_J2C",
    "OS_Dataset", "Rice_Chem"
]

models = ["GPT-4o-mini", "Gemini-2.5-Flash", "LLaMA-4-Scout"]
strategies = ["Ind", "Ded", "Abd", "Ind+Abd", "Ind+Ded", "Ded+Abd"]

# =========================
#   REGRESSION DATA (QWK)
# =========================

reg_pairs = [(d, m) for d in regression_datasets for m in models]

regression_data = {
    "Dataset": [p[0] for p in reg_pairs],
    "Model": [p[1] for p in reg_pairs],
    
    "Ind": [
        # ASAP-SAS
        0.6431, 0.6025, 0.5063,
        # CSEE
        0.5226, 0.6226, 0.6216,
        # Mohlar
        0.0281, 0.1259, 0.0234,
        # Regrading_J2C
        0.2463, -0.20, 0.20,
        # OS_Dataset_Avg
        0.0420, -0.0382, -0.0193,
        # Rice_Chem_Avg
        0.5110, 0.6444, 0.5220,
    ],
    
    "Ded": [
        # ASAP-SAS
        0.5512, 0.6470, 0.6077,
        # CSEE
        0.5112, 0.5053, 0.5902,
        # Mohlar
        0.0172, 0.1452, 0.07,
        # Regrading_J2C
        0.2746, 0.2579, 0.44,
        # OS_Dataset_Avg
        -0.0238, 0.0088, -0.0387,
        # Rice_Chem_Avg
        0.1944, 0.3911, 0.34,
    ],
    
    "Abd": [
        # ASAP-SAS
        0.5973, 0.6262, 0.4991,
        # CSEE
        0.4764, 0.5682, 0.6215,
        # Mohlar
        0.0234, 0.1814, 0.0138,
        # Regrading_J2C
        0.1805, 0.1897, 0.3054,
        # OS_Dataset_Avg
        -0.0434, -0.04, 0.14,
        # Rice_Chem_Avg
        0.1362, 0.3106, 0.2283,
    ],
    
    "Ind+Abd": [
        # ASAP-SAS
        0.6558, 0.6020, 0.4758,
        # CSEE
        0.5050, 0.6419, 0.6133,
        # Mohlar
        0.01, 0.2556, -0.0375,
        # Regrading_J2C
        0.2553, 0.1741, 0.3204,
        # OS_Dataset_Avg
        -0.0462, -0.02, 0.09,
        # Rice_Chem_Avg
        0.4864, 0.5608, 0.3696,
    ],
    
    "Ind+Ded": [
        # ASAP-SAS
        0.6459, 0.5953, 0.5030,
        # CSEE
        0.55, 0.6433, 0.6600,
        # Mohlar
        -0.01, 0.1872, 0.1335,
        # Regrading_J2C
        0.2393, 0.2178, 0.3950,
        # OS_Dataset_Avg
        0.002, 0.008, 0.1727,
        # Rice_Chem_Avg
        0.4406, 0.5680, 0.4469,
    ],
    
    "Ded+Abd": [
        # ASAP-SAS
        0.4914, 0.6520, 0.3565,
        # CSEE
        0.4254, 0.3341, 0.5823,
        # Mohlar
        -0.0064, 0.0710, -0.0565,
        # Regrading_J2C
        0.3569, 0.1807, 0.41,
        # OS_Dataset_Avg
        -0.0954, -0.11, 0.0839,
        # Rice_Chem_Avg
        0.2444, 0.4406, 0.2762,
    ],
}

df_regression = pd.DataFrame(regression_data)

# =========================
#   CREATE HEATMAPS
# =========================

sns.set_theme(style="whitegrid")
cmap = sns.color_palette("YlGnBu", as_cmap=True)

fig, axes = plt.subplots(3, 2, figsize=(12, 10), sharex=True, sharey=True)
axes = axes.flatten()

for ax, dataset in zip(axes, regression_datasets):
    subset = df_regression[df_regression["Dataset"] == dataset].set_index("Model")[strategies]
    
    sns.heatmap(
        subset, annot=True, cmap=cmap, fmt=".2f",
        cbar=False, ax=ax, linewidths=0.5, linecolor="white",
        annot_kws={"size": 10, "weight": "bold"},
        vmin=-0.5, vmax=1.0,
        square=True
    )
    
    # Color text based on value
    for text in ax.texts:
        try:
            val = float(text.get_text())
            text.set_color('white' if val > 0.3 else 'black')
        except:
            pass
    
    ax.set_title(dataset, fontsize=12, weight="bold")
    ax.set_ylabel("")
    ax.tick_params(axis="x", rotation=40, labelsize=9)
    ax.tick_params(axis="y", labelsize=9)

# =========================
#   SHARED COLORBAR
# =========================

cbar_ax = fig.add_axes([0.93, 0.25, 0.02, 0.5])
norm = plt.Normalize(vmin=-0.5, vmax=1.0)
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
cbar = plt.colorbar(sm, cax=cbar_ax)
cbar.set_label("Quadratic Weighted Kappa (QWK)", fontsize=11, weight="bold")

# =========================
#   MAIN TITLE
# =========================

fig.suptitle(
    "ASAG Regression — Model-wise QWK Across Reasoning Strategies",
    fontsize=16, weight="bold", y=0.98
)

plt.subplots_adjust(left=0.08, right=0.91, top=0.94, bottom=0.08, hspace=0.35, wspace=0.25)

plt.savefig("/home/ts1506.UNT/Desktop/Work/ASAG_regression_heatmaps_final.pdf", dpi=400, bbox_inches='tight')
plt.show()