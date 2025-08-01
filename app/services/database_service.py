# app/services/database_service.py (Fixed session management)

from sqlalchemy.orm import Session
from sqlalchemy import func, desc, text
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
import json

from app.models.database import OutputSubmission, Dataset, EvaluationResult, Essay
from app.config.database import get_db_session


class DatabaseService:
    """Service for database operations for BESESR platform (CSV uploads only)"""

    # ============================================================================
    # OUTPUT SUBMISSIONS (CSV file uploads)
    # ============================================================================

    @staticmethod
    def save_output_submission(submission_data: Dict[str, Any]) -> Dict[str, Any]:
        """Save a new dataset output submission (CSV) - Fixed session management"""
        try:
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
                db.flush()  # This assigns the ID
                
                # Extract data we need before session closes
                submission_dict = {
                    "id": submission.id,
                    "dataset_name": submission.dataset_name,
                    "submitter_name": submission.submitter_name,
                    "submitter_email": submission.submitter_email,
                    "file_path": submission.file_path,
                    "file_format": submission.file_format,
                    "status": submission.status,
                    "description": submission.description,
                    "submission_time": submission.submission_time
                }
                
                db.commit()  # Commit the transaction
                print(f"💾 Saved output submission {submission.id} for dataset: {submission.dataset_name}")
                
                return submission_dict
                
        except Exception as e:
            print(f"❌ Error saving submission: {e}")
            raise

    @staticmethod
    def update_output_submission_status(
        submission_id: int,
        status: str,
        evaluation_result: Optional[Dict[str, Any]] = None,
        error_message: Optional[str] = None,
        processing_time: Optional[float] = None
    ):
        """Update status and result of an output submission"""
        try:
            with get_db_session() as db:
                submission = db.query(OutputSubmission).filter(OutputSubmission.id == submission_id).first()
                if submission:
                    submission.status = status
                    if evaluation_result:
                        submission.evaluation_result = json.dumps(evaluation_result)  # Store as JSON string
                    if error_message:
                        submission.error_message = error_message
                    if processing_time:
                        submission.processing_time = processing_time
                    db.commit()
                    print(f"📊 Updated output submission {submission_id} status to {status}")
                else:
                    print(f"⚠️ Output submission {submission_id} not found")
        except Exception as e:
            print(f"❌ Error updating submission status: {e}")
            raise

    @staticmethod
    def get_output_submission(submission_id: int) -> Optional[Dict[str, Any]]:
        """Fetch a single output submission as dictionary"""
        try:
            with get_db_session() as db:
                submission = db.query(OutputSubmission).filter(OutputSubmission.id == submission_id).first()
                if submission:
                    # Parse evaluation_result JSON if it exists
                    evaluation_result = None
                    if submission.evaluation_result:
                        try:
                            evaluation_result = json.loads(submission.evaluation_result)
                        except:
                            evaluation_result = submission.evaluation_result
                    
                    return {
                        "id": submission.id,
                        "dataset_name": submission.dataset_name,
                        "submitter_name": submission.submitter_name,
                        "submitter_email": submission.submitter_email,
                        "file_path": submission.file_path,
                        "file_format": submission.file_format,
                        "status": submission.status,
                        "description": submission.description,
                        "evaluation_result": evaluation_result,
                        "error_message": submission.error_message,
                        "submission_time": submission.submission_time,
                        "processing_time": submission.processing_time
                    }
                return None
        except Exception as e:
            print(f"❌ Error getting submission: {e}")
            return None

    @staticmethod
    def list_output_submissions(dataset_name: Optional[str] = None, limit: int = 50) -> List[Dict[str, Any]]:
        """List recent output submissions (filtered by dataset if provided)"""
        try:
            with get_db_session() as db:
                query = db.query(OutputSubmission)
                if dataset_name:
                    query = query.filter(OutputSubmission.dataset_name == dataset_name)
                
                submissions = query.order_by(desc(OutputSubmission.submission_time)).limit(limit).all()
                
                results = []
                for submission in submissions:
                    # Parse evaluation_result JSON if it exists
                    evaluation_result = None
                    if submission.evaluation_result:
                        try:
                            evaluation_result = json.loads(submission.evaluation_result)
                        except:
                            evaluation_result = submission.evaluation_result
                    
                    results.append({
                        "id": submission.id,
                        "dataset_name": submission.dataset_name,
                        "submitter_name": submission.submitter_name,
                        "submitter_email": submission.submitter_email,
                        "status": submission.status,
                        "evaluation_result": evaluation_result,
                        "submission_time": submission.submission_time,
                        "processing_time": submission.processing_time
                    })
                
                return results
        except Exception as e:
            print(f"❌ Error listing submissions: {e}")
            return []

    @staticmethod
    def get_output_leaderboard(limit: int = 20) -> List[Dict[str, Any]]:
        """Retrieve leaderboard based on output submission results"""
        try:
            with get_db_session() as db:
                # Get submissions with completed status
                submissions = db.query(OutputSubmission).filter(
                    OutputSubmission.status == "completed",
                    OutputSubmission.evaluation_result.isnot(None)
                ).order_by(desc(OutputSubmission.submission_time)).limit(limit * 2).all()  # Get more to sort by QWK

                leaderboard_data = []
                for submission in submissions:
                    if submission.evaluation_result:
                        try:
                            # Parse JSON string to dict
                            if isinstance(submission.evaluation_result, str):
                                metrics = json.loads(submission.evaluation_result)
                            else:
                                metrics = submission.evaluation_result or {}
                            
                            qwk = metrics.get("quadratic_weighted_kappa", 0.0)
                            if qwk > 0:  # Only include submissions with valid scores
                                leaderboard_data.append({
                                    "submission_id": submission.id,
                                    "dataset_name": submission.dataset_name,
                                    "submitter_name": submission.submitter_name,
                                    "submitter_email": submission.submitter_email,
                                    "quadratic_weighted_kappa": qwk,
                                    "pearson_correlation": metrics.get("pearson_correlation", 0.0),
                                    "mean_absolute_error": metrics.get("mean_absolute_error", 999.0),
                                    "accuracy": metrics.get("accuracy", 0.0),
                                    "essays_evaluated": metrics.get("essays_evaluated", 0),
                                    "submission_time": submission.submission_time.isoformat() if submission.submission_time else None,
                                    "submission_type": "csv_upload"
                                })
                        except Exception as e:
                            print(f"⚠️ Error parsing evaluation result for submission {submission.id}: {e}")
                            continue

                # Sort by QWK descending and add ranks
                leaderboard_data.sort(key=lambda x: x["quadratic_weighted_kappa"], reverse=True)
                
                leaderboard = []
                for i, entry in enumerate(leaderboard_data[:limit]):
                    entry["rank"] = i + 1
                    leaderboard.append(entry)

                return leaderboard
        except Exception as e:
            print(f"❌ Error getting leaderboard: {e}")
            return []

    # ============================================================================
    # EVALUATION RESULTS
    # ============================================================================

    @staticmethod
    def save_evaluation_result(submission_id: int, dataset_name: str, result: Dict[str, Any]) -> Dict[str, Any]:
        """Save evaluation result for a submission"""
        try:
            with get_db_session() as db:
                evaluation = EvaluationResult(
                    submission_id=submission_id,
                    dataset_name=dataset_name,
                    quadratic_weighted_kappa=result.get("quadratic_weighted_kappa", 0.0),
                    pearson_correlation=result.get("pearson_correlation", 0.0),
                    spearman_correlation=result.get("spearman_correlation", 0.0),
                    mean_absolute_error=result.get("mean_absolute_error", 999.0),
                    root_mean_squared_error=result.get("root_mean_squared_error", 999.0),
                    f1_score=result.get("f1_score", 0.0),
                    accuracy=result.get("accuracy", 0.0),
                    essays_evaluated=result.get("essays_evaluated", 0),
                    evaluation_duration=result.get("evaluation_duration", 0.0),
                    status=result.get("status", "completed"),
                    detailed_metrics=json.dumps(result.get("detailed_metrics", {})),
                    error_message=result.get("error_message")
                )
                db.add(evaluation)
                db.flush()
                
                evaluation_dict = {
                    "id": evaluation.id,
                    "submission_id": evaluation.submission_id,
                    "dataset_name": evaluation.dataset_name,
                    "quadratic_weighted_kappa": evaluation.quadratic_weighted_kappa,
                    "pearson_correlation": evaluation.pearson_correlation,
                    "status": evaluation.status
                }
                
                db.commit()
                print(f"📊 Saved evaluation result: submission {submission_id} - {dataset_name}")
                return evaluation_dict
        except Exception as e:
            print(f"❌ Error saving evaluation result: {e}")
            raise

    @staticmethod
    def get_evaluation_results_by_submission(submission_id: int) -> List[Dict[str, Any]]:
        """Get all evaluation results for a submission"""
        try:
            with get_db_session() as db:
                evaluations = db.query(EvaluationResult).filter(
                    EvaluationResult.submission_id == submission_id
                ).all()
                
                results = []
                for eval_result in evaluations:
                    # Parse detailed_metrics JSON if it exists
                    detailed_metrics = {}
                    if eval_result.detailed_metrics:
                        try:
                            detailed_metrics = json.loads(eval_result.detailed_metrics)
                        except:
                            detailed_metrics = {}
                    
                    results.append({
                        'id': eval_result.id,
                        'submission_id': eval_result.submission_id,
                        'dataset_name': eval_result.dataset_name,
                        'quadratic_weighted_kappa': eval_result.quadratic_weighted_kappa,
                        'pearson_correlation': eval_result.pearson_correlation,
                        'spearman_correlation': eval_result.spearman_correlation,
                        'mean_absolute_error': eval_result.mean_absolute_error,
                        'root_mean_squared_error': eval_result.root_mean_squared_error,
                        'f1_score': eval_result.f1_score,
                        'accuracy': eval_result.accuracy,
                        'essays_evaluated': eval_result.essays_evaluated,
                        'evaluation_time': eval_result.evaluation_time.isoformat() if eval_result.evaluation_time else None,
                        'evaluation_duration': eval_result.evaluation_duration,
                        'status': eval_result.status,
                        'error_message': eval_result.error_message,
                        'detailed_metrics': detailed_metrics
                    })
                
                return results
        except Exception as e:
            print(f"❌ Error getting evaluation results: {e}")
            return []

    # ============================================================================
    # PLATFORM STATISTICS
    # ============================================================================

    @staticmethod
    def get_platform_stats() -> Dict[str, Any]:
        """Get overall platform statistics"""
        try:
            with get_db_session() as db:
                # Count submissions
                total_submissions = db.query(OutputSubmission).count()
                completed_submissions = db.query(OutputSubmission).filter(
                    OutputSubmission.status == "completed"
                ).count()
                
                # Count evaluations
                total_evaluations = db.query(EvaluationResult).count()
                completed_evaluations = db.query(EvaluationResult).filter(
                    EvaluationResult.status == "completed"
                ).count()
                
                # Count datasets
                total_datasets = db.query(Dataset).count()
                
                return {
                    "total_submissions": total_submissions,
                    "completed_submissions": completed_submissions,
                    "total_evaluations": total_evaluations,
                    "completed_evaluations": completed_evaluations,
                    "total_datasets": total_datasets,
                    "success_rate": (completed_submissions / total_submissions * 100) if total_submissions > 0 else 0
                }
        except Exception as e:
            print(f"❌ Error getting platform stats: {e}")
            return {
                "total_submissions": 0,
                "completed_submissions": 0,
                "total_evaluations": 0,
                "completed_evaluations": 0,
                "total_datasets": 0,
                "success_rate": 0
            }

    @staticmethod
    def get_recent_submissions(limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent submissions for monitoring"""
        try:
            submissions = DatabaseService.list_output_submissions(limit=limit)
            
            results = []
            for submission in submissions:
                eval_score = 0
                if submission.get("evaluation_result"):
                    eval_score = submission["evaluation_result"].get("quadratic_weighted_kappa", 0)
                
                results.append({
                    "id": submission["id"],
                    "dataset_name": submission["dataset_name"],
                    "submitter_name": submission["submitter_name"],
                    "status": submission["status"],
                    "created_at": submission["submission_time"].isoformat() if submission["submission_time"] else None,
                    "evaluation_score": eval_score
                })
            
            return results
        except Exception as e:
            print(f"❌ Error getting recent submissions: {e}")
            return []

    # ============================================================================
    # DATASET MANAGEMENT
    # ============================================================================

    @staticmethod
    def initialize_datasets():
        """Initialize datasets dynamically from HuggingFace configuration"""
        try:
            from app.services.dataset_loader import dataset_manager
            datasets_config = dataset_manager.hf_loader.get_configured_datasets()

            with get_db_session() as db:
                existing_count = db.query(Dataset).count()

                if existing_count == 0:
                    print("🔄 Initializing datasets from HuggingFace configuration...")

                    for dataset_name, config in datasets_config.items():
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

                    db.commit()
                    print(f"✅ Initialized {len(datasets_config)} datasets from HuggingFace")
                else:
                    print(f"✅ Database already has {existing_count} datasets (skipping initialization)")
                    
        except Exception as e:
            print(f"❌ Error initializing datasets: {e}")

    @staticmethod
    def get_all_datasets() -> List[Dict[str, Any]]:
        """Get all datasets from database"""
        try:
            with get_db_session() as db:
                datasets = db.query(Dataset).filter(Dataset.is_active == True).all()
                
                result = []
                for dataset in datasets:
                    result.append({
                        "name": dataset.name,
                        "description": dataset.description,
                        "huggingface_id": dataset.huggingface_id,
                        "essay_count": dataset.essay_count,
                        "avg_essay_length": dataset.avg_essay_length,
                        "score_range": [dataset.score_range_min, dataset.score_range_max],
                        "is_active": dataset.is_active,
                        "created_at": dataset.created_at.isoformat() if dataset.created_at else None
                    })
                
                return result
                
        except Exception as e:
            print(f"❌ Error getting datasets: {e}")
            return []

    @staticmethod
    def update_dataset_stats(dataset_name: str, essay_count: int, avg_essay_length: float):
        """Update dataset statistics after loading"""
        try:
            with get_db_session() as db:
                dataset = db.query(Dataset).filter(Dataset.name == dataset_name).first()
                if dataset:
                    dataset.essay_count = essay_count
                    dataset.avg_essay_length = avg_essay_length
                    db.commit()
                    print(f"📊 Updated stats for {dataset_name}: {essay_count} essays, avg length {avg_essay_length:.1f}")
        except Exception as e:
            print(f"❌ Error updating dataset stats: {e}")

    # ============================================================================
    # UTILITY METHODS
    # ============================================================================

    @staticmethod
    def get_leaderboard(metric: str = 'avg_quadratic_weighted_kappa', limit: int = 20) -> List[Dict[str, Any]]:
        """Get leaderboard based on submission performance (compatibility method)"""
        return DatabaseService.get_output_leaderboard(limit)

    @staticmethod
    def health_check() -> Dict[str, Any]:
        """Check database health"""
        try:
            with get_db_session() as db:
                # Simple query to test connection
                db.execute(text("SELECT 1")).fetchone()
                
                return {
                    "status": "healthy",
                    "timestamp": datetime.now().isoformat(),
                    "database": "connected"
                }
        except Exception as e:
            return {
                "status": "unhealthy",
                "timestamp": datetime.now().isoformat(),
                "database": "disconnected",
                "error": str(e)
            }