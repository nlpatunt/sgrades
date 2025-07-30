from sqlalchemy.orm import Session
from sqlalchemy import func, desc
from typing import List, Optional, Dict, Any
from datetime import datetime

from app.models.database import OutputSubmission, Dataset
from app.config.database import get_db_session
from app.db import get_db_connection



class DatabaseService:
    """Service for database operations for Output Submissions"""

    @staticmethod
    def save_output_submission(submission_data: Dict[str, Any]) -> OutputSubmission:
        """Save a new dataset output submission (CSV or JSON)"""
        with get_db_session() as db:
            submission = OutputSubmission(
                dataset_name=submission_data["dataset_name"],
                submitter_name=submission_data["submitter_name"],
                submitter_email=submission_data["submitter_email"],
                file_path=submission_data["file_path"],
                file_format=submission_data["file_format"],
                status=submission_data.get("status", "submitted"),
                description=submission_data.get("description")
            )
            db.add(submission)
            db.flush()
            print(f"💾 Saved output submission for dataset: {submission.dataset_name}")
            return submission

    @staticmethod
    def update_output_submission_status(
        submission_id: int,
        status: str,
        evaluation_result: Optional[Dict[str, Any]] = None,
        error_message: Optional[str] = None
    ):
        """Update status and result of an output submission"""
        with get_db_session() as db:
            submission = db.query(OutputSubmission).filter(OutputSubmission.id == submission_id).first()
            if submission:
                submission.status = status
                if evaluation_result:
                    submission.evaluation_result = evaluation_result
                if error_message:
                    submission.error_message = error_message
                db.commit()
                print(f"📊 Updated output submission {submission_id} status to {status}")
            else:
                print(f"⚠️ Output submission {submission_id} not found")

    @staticmethod
    def get_output_submission(submission_id: int) -> Optional[OutputSubmission]:
        """Fetch a single output submission"""
        with get_db_session() as db:
            return db.query(OutputSubmission).filter(OutputSubmission.id == submission_id).first()

    @staticmethod
    def list_output_submissions(dataset_name: Optional[str] = None, limit: int = 50) -> List[OutputSubmission]:
        """List recent output submissions (filtered by dataset if provided)"""
        with get_db_session() as db:
            query = db.query(OutputSubmission)
            if dataset_name:
                query = query.filter(OutputSubmission.dataset_name == dataset_name)
            return query.order_by(desc(OutputSubmission.submission_time)).limit(limit).all()

    @staticmethod
    def get_output_leaderboard(limit: int = 20) -> List[Dict[str, Any]]:
        """Retrieve leaderboard based on output submission results"""
        with get_db_session() as db:
            submissions = db.query(OutputSubmission).filter(
                OutputSubmission.status == "completed"
            ).order_by(
                OutputSubmission.evaluation_result["quadratic_weighted_kappa"].as_float().desc()
            ).limit(limit).all()

            leaderboard = []
            for i, sub in enumerate(submissions):
                metrics = sub.evaluation_result or {}
                leaderboard.append({
                    "rank": i + 1,
                    "submission_id": sub.id,
                    "dataset_name": sub.dataset_name,
                    "submitter_name": sub.submitter_name,
                    "submitter_email": sub.submitter_email,
                    "quadratic_weighted_kappa": metrics.get("quadratic_weighted_kappa", 0.0),
                    "pearson_correlation": metrics.get("pearson_correlation", 0.0),
                    "mean_absolute_error": metrics.get("mean_absolute_error", 999.0),
                    "accuracy": metrics.get("accuracy", 0.0),
                    "submission_time": sub.submission_time.isoformat() if sub.submission_time else None
                })

            return leaderboard

    @staticmethod
    def get_output_submission_by_id(submission_id: str) -> Optional[OutputSubmission]:
        with get_db_session() as db:
            return db.query(OutputSubmission).filter(OutputSubmission.id == submission_id).first()

    @staticmethod
    def get_platform_stats():
        conn = get_db_connection()
        cursor = conn.cursor()

        cursor.execute("SELECT COUNT(*) FROM models")
        total_models = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM evaluations")
        total_evaluations = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM datasets")
        total_datasets = cursor.fetchone()[0]

        conn.close()

        return {
            "total_models_submitted": total_models,
            "total_evaluations_completed": total_evaluations,
            "total_datasets": total_datasets
        }

    @staticmethod
    def get_leaderboard(metric='avg_quadratic_weighted_kappa', limit=20):
        conn = get_db_connection()
        cursor = conn.cursor()

        query = f"""
            SELECT 
                m.model_id,
                m.model_name,
                m.submitter_name,
                m.submission_time,
                AVG(e.quadratic_weighted_kappa) AS avg_quadratic_weighted_kappa,
                AVG(e.pearson_correlation) AS avg_pearson_correlation,
                COUNT(e.dataset_id) AS datasets_completed,
                (SELECT COUNT(*) FROM datasets) AS total_datasets
            FROM models m
            LEFT JOIN evaluations e ON m.model_id = e.model_id
            GROUP BY m.model_id
            ORDER BY {metric} DESC
            LIMIT ?
        """
        cursor.execute(query, (limit,))
        rows = cursor.fetchall()

        leaderboard = []
        for idx, row in enumerate(rows, start=1):
            leaderboard.append({
                "rank": idx,
                "model_id": row["model_id"],
                "model_name": row["model_name"],
                "submitter_name": row["submitter_name"],
                "submission_time": row["submission_time"],
                "avg_quadratic_weighted_kappa": row["avg_quadratic_weighted_kappa"],
                "avg_pearson_correlation": row["avg_pearson_correlation"],
                "datasets_completed": row["datasets_completed"],
                "total_datasets": row["total_datasets"]
            })

        conn.close()
        return leaderboard


    @staticmethod
    def initialize_datasets():
        """Initialize datasets dynamically from HuggingFace configuration"""
        from app.services.dataset_loader import dataset_manager
        datasets_config = dataset_manager.hf_loader.get_configured_datasets()

        with get_db_session() as db:
            existing_count = db.query(Dataset).count()

            if existing_count == 0:
                print("🔄 Initializing datasets from HuggingFace configuration...")

                for dataset_name, config in datasets_config.items():
                    """hf_info = dataset_manager.hf_loader.get_dataset_info(config["huggingface_id"])"""
                    dataset = Dataset(
                        name=dataset_name,
                        description=config["description"],
                        huggingface_id=config["huggingface_id"],
                        essay_count=0,
                        avg_essay_length=0.0,
                        score_range_min=config["score_range"][0],
                        score_range_max=config["score_range"][1],
                        is_active=True
                    )
                    db.add(dataset)
                    print(f"   ✅ Added dataset: {dataset_name} -> {config['huggingface_id']}")

                print(f"✅ Initialized {len(datasets_config)} datasets from HuggingFace")
            else:
                print(f"✅ Database already has {existing_count} datasets (skipping initialization)")
