import numpy as np
from sklearn.metrics import cohen_kappa_score, mean_absolute_error, mean_squared_error, f1_score
from scipy.stats import pearsonr, spearmanr
from typing import List, Dict, Any


def calculate_evaluation_metrics(matched_data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Compute evaluation metrics for essay score predictions from matched data.
    """
    
    if not matched_data or len(matched_data) == 0:
        return {
            'quadratic_weighted_kappa': 0.0,
            'pearson_correlation': 0.0,
            'spearman_correlation': 0.0,
            'mean_absolute_error': 999.0,
            'root_mean_squared_error': 999.0,
            'f1_score': 0.0,
            'accuracy': 0.0,
            'essays_evaluated': 0
        }

    try:
        # Extract lists from matched_data
        y_true = [item['human_score'] for item in matched_data]
        y_pred = [item['predicted_score'] for item in matched_data]
        
        # Convert to numpy arrays
        true_arr = np.array(y_true)
        pred_arr = np.array(y_pred)

        # Classification rounding
        true_int = np.round(true_arr).astype(int)
        pred_int = np.round(pred_arr).astype(int)

        # --- Metric Calculations ---
        # QWK
        qwk = cohen_kappa_score(true_int, pred_int, weights="quadratic")

        # Pearson and Spearman
        pearson_r, _ = pearsonr(true_arr, pred_arr)
        pearson_r = 0.0 if np.isnan(pearson_r) else pearson_r

        spearman_r, _ = spearmanr(true_arr, pred_arr)
        spearman_r = 0.0 if np.isnan(spearman_r) else spearman_r

        # MAE and RMSE
        mae = mean_absolute_error(true_arr, pred_arr)
        rmse = np.sqrt(mean_squared_error(true_arr, pred_arr))

        # F1 and Accuracy (based on rounded predictions)
        f1 = f1_score(true_int, pred_int, average="weighted", zero_division=0)
        accuracy = np.mean(true_int == pred_int)

        return {
            "quadratic_weighted_kappa": round(float(qwk), 3),
            "pearson_correlation": round(float(pearson_r), 3),
            "spearman_correlation": round(float(spearman_r), 3),
            "mean_absolute_error": round(float(mae), 3),
            "root_mean_squared_error": round(float(rmse), 3),
            "f1_score": round(float(f1), 3),
            "accuracy": round(float(accuracy), 3),
            "essays_evaluated": len(matched_data)
        }

    except Exception as e:
        print(f"❌ Error calculating metrics: {e}")
        return {
            "quadratic_weighted_kappa": 0.0,
            "pearson_correlation": 0.0,
            "spearman_correlation": 0.0,
            "mean_absolute_error": 999.0,
            "root_mean_squared_error": 999.0,
            "f1_score": 0.0,
            "accuracy": 0.0,
            "essays_evaluated": 0
        }