# app/api/routes/leaderboard.py
from fastapi import APIRouter, HTTPException
from typing import List, Dict, Any, Optional
from datetime import datetime
import statistics

from app.services.database_service import DatabaseService
from app.models.pydantic_models import (
    CompleteLeaderboardEntry, PlatformStats, HealthCheck, 
    LeaderboardEntry, ResearcherProgress, AvailableMetrics, MetricInfo
)

router = APIRouter(prefix="/leaderboard", tags=["leaderboard"])

@router.get("/", response_model=List[CompleteLeaderboardEntry])
async def get_leaderboard(metric: str = "avg_quadratic_weighted_kappa", limit: int = 20):
    """Get leaderboard rankings for complete 15-dataset benchmarks only"""
    
    try:
        print(f"🏆 Fetching complete benchmark leaderboard with metric: {metric}, limit: {limit}")
        
        # Ensure metric has avg_ prefix
        if not metric.startswith("avg_"):
            metric = f"avg_{metric}"
        
        # Get complete benchmark leaderboard from database
        leaderboard_data = DatabaseService.get_complete_benchmark_leaderboard(metric=metric, limit=limit)
        
        if not leaderboard_data:
            print("📋 No complete benchmarks found - returning empty leaderboard")
            return []
        
        print(f"📊 Retrieved {len(leaderboard_data)} complete benchmarks from database")
        
        # Convert to Pydantic models
        leaderboard_entries = []
        for entry in leaderboard_data:
            leaderboard_entry = CompleteLeaderboardEntry(
                rank=entry.get('rank', 0),
                submitter_name=entry.get('submitter_name', 'Unknown'),
                submitter_email=entry.get('submitter_email'),
                total_datasets=entry.get('total_datasets', 15),
                total_essays_evaluated=entry.get('total_essays_evaluated', 0),
                avg_quadratic_weighted_kappa=entry.get('avg_quadratic_weighted_kappa', 0.0),
                avg_pearson_correlation=entry.get('avg_pearson_correlation', 0.0),
                avg_mean_absolute_error=entry.get('avg_mean_absolute_error', 999.0),
                avg_f1_score=entry.get('avg_f1_score', 0.0),
                avg_accuracy=entry.get('avg_accuracy', 0.0),
                datasets_completed=entry.get('datasets_completed', []),
                completion_rate=entry.get('completion_rate', 0.0),
                benchmark_type=entry.get('benchmark_type', 'complete_15_dataset')
            )
            leaderboard_entries.append(leaderboard_entry)
        
        # Log top performer for debugging
        if leaderboard_entries:
            top_model = leaderboard_entries[0]
            print(f"🥇 Top performer: {top_model.submitter_name} (Avg QWK: {top_model.avg_quadratic_weighted_kappa})")
        
        return leaderboard_entries
        
    except Exception as e:
        print(f"❌ Error getting complete benchmark leaderboard: {e}")
        import traceback
        traceback.print_exc()
        return []

@router.get("/stats", response_model=PlatformStats)
async def get_leaderboard_stats():
    """Get platform statistics"""
    
    try:
        print("📈 Fetching platform statistics...")
        
        stats = DatabaseService.get_platform_stats()
        
        # Get benchmark-specific stats
        complete_benchmarks = DatabaseService.get_complete_benchmark_leaderboard(limit=100)
        individual_submissions = DatabaseService.get_output_leaderboard(limit=100)
        
        # Create datasets_available list
        try:
            from app.services.dataset_loader import dataset_manager
            datasets_available = list(dataset_manager.datasets_config.keys())
        except:
            datasets_available = ["ASAP-AES", "ASAP-SAS", "ASAP2", "rice_chem", "CSEE", "EFL", 
                                "grade_like_a_human_dataset_os", "persuade_2", "SciEntSBank", 
                                "BEEtlE", "automatic_short_answer_grading_mohlar"]
        
        enhanced_stats = PlatformStats(
            total_complete_benchmarks=len(complete_benchmarks),
            total_individual_submissions=len(individual_submissions),
            total_datasets=15,
            total_models_submitted=stats.get("total_models_submitted", 0),
            total_evaluations_completed=stats.get("total_evaluations_completed", 0),
            researchers_with_complete_benchmarks=len(complete_benchmarks),
            researchers_in_progress=max(0, stats.get("total_models_submitted", 0) - len(complete_benchmarks)),
            top_performer=complete_benchmarks[0].get('submitter_name') if complete_benchmarks else None,
            latest_submission=stats.get("latest_submission"),
            avg_evaluation_time="15-30 minutes",
            datasets_available=datasets_available,
            platform_uptime_days=stats.get("platform_uptime_days", 1),
            database_status="connected",
            api_version="1.0.0",
            evaluation_success_rate=0.95,
            benchmark_type="complete_15_dataset_only"
        )
        
        print(f"📊 Platform stats: {enhanced_stats.total_models_submitted} researchers, {enhanced_stats.total_complete_benchmarks} complete benchmarks")
        
        return enhanced_stats
        
    except Exception as e:
        print(f"❌ Error getting platform statistics: {e}")
        import traceback
        traceback.print_exc()
        
        return PlatformStats(
            total_complete_benchmarks=0,
            total_individual_submissions=0,
            total_datasets=15,
            total_models_submitted=0,
            total_evaluations_completed=0,
            researchers_with_complete_benchmarks=0,
            researchers_in_progress=0,
            datasets_available=[],
            platform_uptime_days=1,
            database_status="error"
        )

@router.get("/health", response_model=HealthCheck)
async def leaderboard_health():
    """Health check for leaderboard service"""
    
    try:
        stats = DatabaseService.get_platform_stats()
        complete_benchmarks = len(DatabaseService.get_complete_benchmark_leaderboard(limit=1))
        
        return HealthCheck(
            status="healthy",
            service="leaderboard",
            database_connection="ok",
            timestamp=datetime.now().isoformat(),
            complete_benchmarks_available=complete_benchmarks,
            models_available=stats.get("total_models_submitted", 0),
            evaluations_completed=stats.get("total_evaluations_completed", 0)
        )
        
    except Exception as e:
        return HealthCheck(
            status="unhealthy",
            service="leaderboard", 
            database_connection="error",
            timestamp=datetime.now().isoformat(),
            error=str(e)
        )

@router.get("/individual", response_model=List[LeaderboardEntry])
async def get_individual_leaderboard(metric: str = "quadratic_weighted_kappa", limit: int = 20):
    """Get leaderboard for individual dataset submissions (testing only)"""
    
    try:
        print(f"📊 Fetching individual dataset leaderboard (testing)")
        
        # Use the old individual leaderboard method
        leaderboard_data = DatabaseService.get_output_leaderboard(limit=limit)
        
        # Convert to Pydantic models
        leaderboard_entries = []
        for entry in leaderboard_data:
            leaderboard_entry = LeaderboardEntry(
                rank=entry.get('rank', 0),
                submitter_name=entry.get('submitter_name', 'Unknown'),
                submitter_email=entry.get('submitter_email'),
                model_name=entry.get('model_name'),
                quadratic_weighted_kappa=entry.get('quadratic_weighted_kappa', 0.0),
                pearson_correlation=entry.get('pearson_correlation', 0.0),
                mean_absolute_error=entry.get('mean_absolute_error', 999.0),
                accuracy=entry.get('accuracy', 0.0),
                essays_evaluated=entry.get('essays_evaluated', 0),
                submission_time=entry.get('submission_time'),
                submission_type=entry.get('submission_type', 'individual_testing')
            )
            leaderboard_entries.append(leaderboard_entry)
        
        return leaderboard_entries
        
    except Exception as e:
        print(f"❌ Error getting individual leaderboard: {e}")
        return []

@router.get("/progress/{submitter_name}", response_model=ResearcherProgress)
async def get_researcher_progress(submitter_name: str):
    """Get progress for a specific researcher"""
    
    try:
        progress_data = DatabaseService.get_researcher_progress(submitter_name)
        
        return ResearcherProgress(
            submitter_name=progress_data.get("submitter_name", submitter_name),
            completed_datasets=progress_data.get("completed_datasets", 0),
            total_datasets=progress_data.get("total_datasets", 15),
            completion_percentage=progress_data.get("completion_percentage", 0.0),
            is_complete=progress_data.get("is_complete", False),
            completed_dataset_names=progress_data.get("completed_dataset_names", []),
            remaining_datasets=progress_data.get("remaining_datasets", 15)
        )
        
    except Exception as e:
        print(f"❌ Error getting researcher progress: {e}")
        return ResearcherProgress(
            submitter_name=submitter_name,
            completed_datasets=0,
            total_datasets=15,
            completion_percentage=0.0,
            is_complete=False,
            completed_dataset_names=[],
            remaining_datasets=15
        )

@router.get("/metrics", response_model=AvailableMetrics)
async def get_available_metrics():
    """Get list of available evaluation metrics"""
    
    primary_metrics = [
        MetricInfo(
            name="quadratic_weighted_kappa",
            display_name="Quadratic Weighted Kappa",
            description="Agreement measure between model and human scores with quadratic weights",
            range=[0.0, 1.0],
            higher_is_better=True,
            default=True
        ),
        MetricInfo(
            name="pearson_correlation", 
            display_name="Pearson Correlation",
            description="Linear correlation coefficient between model and human scores",
            range=[-1.0, 1.0],
            higher_is_better=True,
            default=False
        )
    ]
    
    secondary_metrics = [
        MetricInfo(
            name="mean_absolute_error",
            display_name="Mean Absolute Error", 
            description="Average absolute difference between model and human scores",
            range=[0.0, "max_score_range"],
            higher_is_better=False
        ),
        MetricInfo(
            name="f1_score",
            display_name="F1 Score",
            description="Harmonic mean of precision and recall for score categories",
            range=[0.0, 1.0], 
            higher_is_better=True
        ),
        MetricInfo(
            name="accuracy",
            display_name="Accuracy",
            description="Percentage of exactly matching scores (rounded)",
            range=[0.0, 1.0],
            higher_is_better=True
        )
    ]
    
    return AvailableMetrics(
        primary_metrics=primary_metrics,
        secondary_metrics=secondary_metrics,
        supported_sorting=["quadratic_weighted_kappa", "pearson_correlation", "mean_absolute_error"]
    )