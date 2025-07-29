import numpy as np
from sklearn.metrics import cohen_kappa_score, mean_absolute_error, mean_squared_error, f1_score
from scipy.stats import pearsonr, spearmanr
from typing import List, Dict, Any


def calculate_evaluation_metrics(y_true: List[float], y_pred: List[float]) -> Dict[str, Any]:
    """
    Compute evaluation metrics for essay score predictions.
    Supports regression-based outputs (float) which are rounded for classification metrics.
    """

    if len(y_true) != len(y_pred) or len(y_true) == 0:
        return {
            'qwk': 0.0,
            'pearson': 0.0,
            'spearman': 0.0,
            'mae': 999.0,
            'rmse': 999.0,
            'f1': 0.0,
            'accuracy': 0.0,
            'distribution': {}
        }

    try:
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
            "qwk": round(float(qwk), 3),
            "pearson": round(float(pearson_r), 3),
            "spearman": round(float(spearman_r), 3),
            "mae": round(float(mae), 3),
            "rmse": round(float(rmse), 3),
            "f1": round(float(f1), 3),
            "accuracy": round(float(accuracy), 3),
            "distribution": {
                "human_mean": round(float(np.mean(true_arr)), 2),
                "model_mean": round(float(np.mean(pred_arr)), 2),
                "human_std": round(float(np.std(true_arr)), 2),
                "model_std": round(float(np.std(pred_arr)), 2)
            }
        }

    except Exception as e:
        print(f"❌ Error calculating metrics: {e}")
        return {
            "qwk": 0.0,
            "pearson": 0.0,
            "spearman": 0.0,
            "mae": 999.0,
            "rmse": 999.0,
            "f1": 0.0,
            "accuracy": 0.0,
            "distribution": {}
        }
