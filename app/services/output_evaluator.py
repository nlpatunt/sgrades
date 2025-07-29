import os
import pandas as pd
from datetime import datetime
from app.services.dataset_loader import dataset_manager
from app.utils.metrics import calculate_evaluation_metrics
from app.services.database_service import DatabaseService


def evaluate_output_submission(submission_id: int, dataset_name: str, file_path: str, file_format: str):
    """
    Evaluate a submitted output file (CSV or JSON) for a given dataset.
    """
    try:
        print(f"📂 Evaluating submission {submission_id} for dataset '{dataset_name}'")

        # Load the submission file
        if file_format == "csv":
            df = pd.read_csv(file_path)
        elif file_format == "json":
            df = pd.read_json(file_path, orient="records", lines=True)
        else:
            raise ValueError(f"Unsupported file format: {file_format}")

        # Validate required columns
        required_columns = {"essay_id", "predicted_score"}
        if not required_columns.issubset(df.columns):
            raise ValueError("File must contain columns: 'essay_id' and 'predicted_score'")

        # Load ground truth
        ground_truth = dataset_manager.load_ground_truth_scores(dataset_name)
        if not ground_truth:
            raise ValueError(f"Ground truth not available for dataset '{dataset_name}'")

        gt_df = pd.DataFrame(ground_truth)  # Must have essay_id, human_score, score_range

        # Merge predictions with ground truth
        merged = pd.merge(df, gt_df, on="essay_id", how="inner")
        if merged.empty:
            raise ValueError("No matching essay_ids found between submission and ground truth")

        # Normalize predicted scores to match ground truth scale
        merged["normalized_pred"] = merged.apply(
            lambda row: normalize_score(
                row["predicted_score"],
                from_range=(1, 6),  # assumed prediction scale
                to_range=tuple(row["score_range"])
            ),
            axis=1
        )
        merged["normalized_true"] = merged["human_score"]

        # Compute evaluation metrics
        metrics = calculate_evaluation_metrics(
            y_true=merged["normalized_true"].tolist(),
            y_pred=merged["normalized_pred"].tolist()
        )

        result = {
            "quadratic_weighted_kappa": metrics.get("qwk"),
            "pearson_correlation": metrics.get("pearson"),
            "spearman_correlation": metrics.get("spearman"),
            "mean_absolute_error": metrics.get("mae"),
            "root_mean_squared_error": metrics.get("rmse"),
            "f1_score": metrics.get("f1"),
            "accuracy": metrics.get("accuracy"),
            "essays_evaluated": len(merged),
            "evaluation_duration": 0,
            "status": "completed",
            "detailed_metrics": {
                "essay_results": merged.to_dict(orient="records"),
                "score_distribution": metrics.get("distribution", {})
            }
        }

        # Save results to the database
        DatabaseService.update_output_submission_status(
            submission_id=submission_id,
            status="completed",
            evaluation_result=result
        )

        print(f"✅ Evaluation completed successfully for submission {submission_id}")
    except Exception as e:
        print(f"❌ Evaluation failed for submission {submission_id}: {e}")
        DatabaseService.update_output_submission_status(
            submission_id=submission_id,
            status="failed",
            error_message=str(e)
        )


def normalize_score(score, from_range, to_range):
    """
    Normalize a score from one scale to another.
    """
    try:
        if from_range == to_range:
            return score
        from_min, from_max = from_range
        to_min, to_max = to_range
        scaled = (score - from_min) / (from_max - from_min)
        return scaled * (to_max - to_min) + to_min
    except Exception:
        return score
