import numpy as np
import pandas as pd

def calculate_stability_metrics(runs_data, model_name):
    """
    Calculate mean, std, and std% for QWK, MAE, RMSE
    
    Parameters:
    -----------
    runs_data : dict
        Dictionary with 'QWK', 'MAE', 'RMSE' keys, each containing list of values
    model_name : str
        Name of the model
    
    Returns:
    --------
    dict : Statistics including mean, std, and std%
    """
    
    results = {}
    
    for metric in ['QWK', 'MAE', 'RMSE']:
        values = np.array(runs_data[metric])
        mean = np.mean(values)
        std = np.std(values, ddof=0)  # Population std (divide by N)
        std_pct = (std / mean) * 100
        
        results[metric] = {
            'values': values,
            'mean': mean,
            'std': std,
            'std_pct': std_pct
        }
    
    # Determine stability category based on QWK std%
    qwk_std_pct = results['QWK']['std_pct']
    if qwk_std_pct < 1.0:
        stability = "VERY STABLE"
    elif qwk_std_pct < 2.0:
        stability = "STABLE"
    elif qwk_std_pct < 3.0:
        stability = "MODERATE"
    else:
        stability = "UNSTABLE"
    
    return results, stability

# =========================
#   DATA INPUT
# =========================

# LLaMA-4-Scout (Ind+Ded)
llama_data = {
    'QWK': [0.9256, 0.9201, 0.9446],
    'MAE': [1.8498, 1.9653, 1.7604],
    'RMSE': [3.5913, 3.7091, 3.1888]
}

# GPT-4o-mini (Ind+Ded)
gpt_data = {
    'QWK': [0.9642, 0.9087, 0.9448],
    'MAE': [1.2912, 2.0462, 1.7704],
    'RMSE': [2.3359, 3.9314, 3.2436]
}

# Gemini-2.5-Flash (Ded)
gemini_data = {
    'QWK': [0.9320, 0.922987400021825, 0.92277153342553],
    'MAE': [1.8970, 1.99768875192604, 1.99922958397535],
    'RMSE': [3.4170, 3.6915970879949, 3.6813568544324]
}

seeds = [42, 123, 456]

# =========================
#   CALCULATE STATISTICS
# =========================

llama_stats, llama_stability = calculate_stability_metrics(llama_data, "LLaMA-4-Scout")
gpt_stats, gpt_stability = calculate_stability_metrics(gpt_data, "GPT-4o-mini")
gemini_stats, gemini_stability = calculate_stability_metrics(gemini_data, "Gemini-2.5-Flash")

# =========================
#   PRINT RESULTS
# =========================

print("=" * 80)
print("RANDOM SAMPLING STABILITY ANALYSIS - ASAP-AES")
print("=" * 80)
print()

models_data = [
    ("LLaMA-4-Scout", "Ind+Ded", llama_data, llama_stats, llama_stability),
    ("GPT-4o-mini", "Ind+Ded", gpt_data, gpt_stats, gpt_stability),
    ("Gemini-2.5-Flash", "Ded", gemini_data, gemini_stats, gemini_stability)
]

for model_name, approach, data, stats, stability in models_data:
    print("-" * 80)
    print(f"MODEL: {model_name}")
    print(f"Approach: {approach}")
    print(f"Training Examples: 5 per run (zero-shot for Gemini)")
    print(f"Seeds: {seeds}")
    print("-" * 80)
    
    # Individual runs
    print("\nINDIVIDUAL RUN RESULTS:")
    for i, seed in enumerate(seeds):
        print(f"  Run {i+1} (seed={seed}):")
        print(f"    QWK:  {data['QWK'][i]:.4f}")
        print(f"    MAE:  {data['MAE'][i]:.4f}")
        print(f"    RMSE: {data['RMSE'][i]:.4f}")
    
    # Statistical summary
    print("\nSTATISTICAL SUMMARY:")
    print(f"  QWK:  Mean={stats['QWK']['mean']:.4f}, Std={stats['QWK']['std']:.4f}, Std%={stats['QWK']['std_pct']:.2f}%")
    print(f"  MAE:  Mean={stats['MAE']['mean']:.4f}, Std={stats['MAE']['std']:.4f}, Std%={stats['MAE']['std_pct']:.2f}%")
    print(f"  RMSE: Mean={stats['RMSE']['mean']:.4f}, Std={stats['RMSE']['std']:.4f}, Std%={stats['RMSE']['std_pct']:.2f}%")
    print(f"\n  Stability Assessment: {stability}")
    print()

# =========================
#   CREATE COMPARISON TABLE
# =========================

print("=" * 80)
print("COMPARATIVE SUMMARY TABLE")
print("=" * 80)

comparison_data = []
for model_name, approach, data, stats, stability in models_data:
    comparison_data.append({
        'Model': model_name,
        'Approach': approach,
        'QWK_Mean': stats['QWK']['mean'],
        'QWK_Std': stats['QWK']['std'],
        'QWK_Std%': stats['QWK']['std_pct'],
        'MAE_Mean': stats['MAE']['mean'],
        'MAE_Std': stats['MAE']['std'],
        'MAE_Std%': stats['MAE']['std_pct'],
        'RMSE_Mean': stats['RMSE']['mean'],
        'RMSE_Std': stats['RMSE']['std'],
        'RMSE_Std%': stats['RMSE']['std_pct'],
        'Stability': stability
    })

df_comparison = pd.DataFrame(comparison_data)
print(df_comparison.to_string(index=False))

# Save to CSV
output_file = "/home/ts1506.UNT/Desktop/Work/randomization_stability_results.csv"
df_comparison.to_csv(output_file, index=False)
print(f"\n✅ Saved comparison table to: {output_file}")

# =========================
#   KEY FINDINGS
# =========================

print("\n" + "=" * 80)
print("KEY FINDINGS")
print("=" * 80)

# Find most stable model
qwk_std_pcts = [(name, stats['QWK']['std_pct']) for name, _, _, stats, _ in models_data]
most_stable = min(qwk_std_pcts, key=lambda x: x[1])
least_stable = max(qwk_std_pcts, key=lambda x: x[1])

print(f"\n1. MOST STABLE: {most_stable[0]} (QWK Std%: {most_stable[1]:.2f}%)")
print(f"2. LEAST STABLE: {least_stable[0]} (QWK Std%: {least_stable[1]:.2f}%)")

# Best performance
best_qwk = max(models_data, key=lambda x: x[3]['QWK']['mean'])
print(f"\n3. BEST MEAN PERFORMANCE: {best_qwk[0]} (QWK: {best_qwk[3]['QWK']['mean']:.4f})")

# Stability ranking
print("\n4. STABILITY RANKING (by QWK Std%):")
sorted_models = sorted(qwk_std_pcts, key=lambda x: x[1])
for i, (name, std_pct) in enumerate(sorted_models, 1):
    print(f"   {i}. {name}: {std_pct:.2f}%")

print("\n" + "=" * 80)