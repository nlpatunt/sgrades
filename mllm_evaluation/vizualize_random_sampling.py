import matplotlib.pyplot as plt
import numpy as np

# Data
models = ['GPT-4o-mini', 'Gemini-2.5-Flash', 'LLaMA-4-Scout']
approaches = ['Ind+Ded', 'Ded', 'Ind+Ded']

# QWK data
qwk_means = [0.9392, 0.9256, 0.9301]
qwk_stds = [0.0230, 0.0043, 0.0105]

# MAE data
mae_means = [1.7026, 1.9652, 1.8585]
mae_stds = [0.3119, 0.0541, 0.0839]

# Colors
colors = ['#e74c3c', '#3498db', '#2ecc71']

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# QWK Plot
ax1 = axes[0]
bars1 = ax1.bar(models, qwk_means, yerr=qwk_stds, capsize=8, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
ax1.set_ylabel('Quadratic Weighted Kappa (QWK)', fontsize=12, weight='bold')
ax1.set_title('QWK Stability', fontsize=13, weight='bold')
ax1.set_ylim([0.88, 0.98])
ax1.grid(axis='y', alpha=0.3, linestyle='--')
ax1.tick_params(axis='x', rotation=15, labelsize=10)

# Add std% labels
for i, (bar, std_pct) in enumerate(zip(bars1, [2.45, 0.46, 1.13])):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height + qwk_stds[i] + 0.003,
             f'Std%: {std_pct:.2f}%',
             ha='center', va='bottom', fontsize=9, weight='bold')

# MAE Plot
ax2 = axes[1]
bars2 = ax2.bar(models, mae_means, yerr=mae_stds, capsize=8, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
ax2.set_ylabel('Mean Absolute Error (MAE)', fontsize=12, weight='bold')
ax2.set_title('MAE Stability', fontsize=13, weight='bold')
ax2.set_ylim([1.0, 2.5])
ax2.grid(axis='y', alpha=0.3, linestyle='--')
ax2.tick_params(axis='x', rotation=15, labelsize=10)

# Add std% labels
for i, (bar, std_pct) in enumerate(zip(bars2, [18.32, 2.75, 4.52])):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height + mae_stds[i] + 0.05,
             f'Std%: {std_pct:.1f}%',
             ha='center', va='bottom', fontsize=9, weight='bold')

fig.suptitle('Random Sampling Stability Analysis — ASAP-AES', fontsize=14, weight='bold', y=0.99)
plt.tight_layout()

# Save as PDF
plt.savefig('/home/ts1506.UNT/Desktop/Work/stability_analysis.pdf', 
            format='pdf',
            bbox_inches='tight',
            dpi=300)

plt.show()

print("✅ Saved: stability_analysis.pdf")