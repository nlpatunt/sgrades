# app/api/routes/leaderboard.py
from fastapi import APIRouter, HTTPException
from typing import List, Dict, Any, Optional
from datetime import datetime
import statistics

from app.services.database_service import DatabaseService

router = APIRouter(prefix="/leaderboard", tags=["leaderboard"])

@router.get("/")
async def get_leaderboard(metric: str = "avg_quadratic_weighted_kappa", limit: int = 20):
    """Get leaderboard rankings for complete 15-dataset benchmarks only"""
    
    try:
        print(f"🏆 Fetching complete benchmark leaderboard with metric: {metric}, limit: {limit}")
        
        # Ensure metric has avg_ prefix
        if not metric.startswith("avg_"):
            metric = f"avg_{metric}"
        
        # Get complete benchmark leaderboard from database
        leaderboard = DatabaseService.get_complete_benchmark_leaderboard(metric=metric, limit=limit)
        
        if not leaderboard:
            print("📋 No complete benchmarks found - returning empty leaderboard")
            return []
        
        print(f"📊 Retrieved {len(leaderboard)} complete benchmarks from database")
        
        # Log top performer for debugging
        if leaderboard:
            top_model = leaderboard[0]
            model_name = top_model.get('submitter_name', 'Unknown Model')
            qwk_score = top_model.get('avg_quadratic_weighted_kappa', 0.0)
            print(f"🥇 Top performer: {model_name} (Avg QWK: {qwk_score})")
        
        return leaderboard
        
    except Exception as e:
        print(f"❌ Error getting complete benchmark leaderboard: {e}")
        import traceback
        traceback.print_exc()
        return []

@router.get("/stats")
async def get_leaderboard_stats():
    """Get platform statistics"""
    
    try:
        print("📈 Fetching platform statistics...")
        
        stats = DatabaseService.get_platform_stats()
        
        # Get benchmark-specific stats
        complete_benchmarks = DatabaseService.get_complete_benchmark_leaderboard(limit=100)
        individual_submissions = DatabaseService.get_output_leaderboard(limit=100)
        
        enhanced_stats = {
            # Core statistics
            "total_complete_benchmarks": len(complete_benchmarks),
            "total_individual_submissions": len(individual_submissions),
            "total_datasets": 15,
            "total_models_submitted": stats.get("total_models_submitted", 0),
            "total_evaluations_completed": stats.get("total_evaluations_completed", 0),
            
            # Benchmark progress
            "researchers_with_complete_benchmarks": len(complete_benchmarks),
            "researchers_in_progress": max(0, stats.get("total_models_submitted", 0) - len(complete_benchmarks)),
            
            # Performance statistics
            "top_performer": complete_benchmarks[0].get('submitter_name') if complete_benchmarks else None,
            "latest_submission": stats.get("latest_submission"),
            "avg_evaluation_time": "15-30 minutes",  # For 15 datasets
            
            # Dataset information
            "datasets_available": [
                "ASAP-AES", "ASAP-SAS", "ASAP2", "ASAP_plus_plus", "rice_chem", 
                "CSEE", "EFL", "grade_like_a_human_dataset_os", "persuade_2",
                "SciEntSBank", "BEEtlE", "automatic_short_answer_grading_mohlar",
                "dataset_13", "dataset_14", "dataset_15"  # Update these with real names
            ],
            
            # Platform metrics
            "platform_uptime_days": stats.get("platform_uptime_days", 1),
            "database_status": "connected",
            "api_version": "1.0.0",
            "evaluation_success_rate": 0.95,
            "benchmark_type": "complete_15_dataset_only"
        }
        
        print(f"📊 Platform stats: {enhanced_stats['total_models_submitted']} researchers, {enhanced_stats['total_complete_benchmarks']} complete benchmarks")
        
        return enhanced_stats
        
    except Exception as e:
        print(f"❌ Error getting platform statistics: {e}")
        import traceback
        traceback.print_exc()
        
        return {
            "total_complete_benchmarks": 0,
            "total_individual_submissions": 0,
            "total_datasets": 15,
            "total_models_submitted": 0,
            "database_status": "error",
            "error": str(e)
        }

@router.get("/health")
async def leaderboard_health():
    """Health check for leaderboard service"""
    
    try:
        stats = DatabaseService.get_platform_stats()
        complete_benchmarks = len(DatabaseService.get_complete_benchmark_leaderboard(limit=1))
        
        return {
            "status": "healthy",
            "service": "leaderboard",
            "database_connection": "ok",
            "complete_benchmarks_available": complete_benchmarks,
            "models_available": stats.get("total_models_submitted", 0),
            "evaluations_completed": stats.get("total_evaluations_completed", 0),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        return {
            "status": "unhealthy",
            "service": "leaderboard", 
            "database_connection": "error",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

@router.get("/individual")
async def get_individual_leaderboard(metric: str = "quadratic_weighted_kappa", limit: int = 20):
    """Get leaderboard for individual dataset submissions (testing only)"""
    
    try:
        print(f"📊 Fetching individual dataset leaderboard (testing)")
        
        # Use the old individual leaderboard method
        leaderboard = DatabaseService.get_output_leaderboard(limit=limit)
        
        return {
            "leaderboard": leaderboard,
            "note": "This shows individual dataset submissions for testing only. Only complete 15-dataset benchmarks appear on the main leaderboard."
        }
        
    except Exception as e:
        print(f"❌ Error getting individual leaderboard: {e}")
        return []

@router.get("/progress/{submitter_name}")
async def get_researcher_progress(submitter_name: str):
    """Get progress for a specific researcher"""
    
    try:
        progress = DatabaseService.get_researcher_progress(submitter_name)
        return progress
        
    except Exception as e:
        print(f"❌ Error getting researcher progress: {e}")
        return {
            "submitter_name": submitter_name,
            "completed_datasets": 0,
            "total_datasets": 15,
            "is_complete": False,
            "error": str(e)
        }

@router.get("/metrics")
async def get_available_metrics():
    """Get list of available evaluation metrics"""
    
    return {
        "primary_metrics": [
            {
                "name": "quadratic_weighted_kappa",
                "display_name": "Quadratic Weighted Kappa",
                "description": "Agreement measure between model and human scores with quadratic weights",
                "range": [0.0, 1.0],
                "higher_is_better": True,
                "default": True
            },
            {
                "name": "pearson_correlation", 
                "display_name": "Pearson Correlation",
                "description": "Linear correlation coefficient between model and human scores",
                "range": [-1.0, 1.0],
                "higher_is_better": True,
                "default": False
            }
        ],
        "secondary_metrics": [
            {
                "name": "mean_absolute_error",
                "display_name": "Mean Absolute Error", 
                "description": "Average absolute difference between model and human scores",
                "range": [0.0, "max_score_range"],
                "higher_is_better": False
            },
            {
                "name": "f1_score",
                "display_name": "F1 Score",
                "description": "Harmonic mean of precision and recall for score categories",
                "range": [0.0, 1.0], 
                "higher_is_better": True
            },
            {
                "name": "accuracy",
                "display_name": "Accuracy",
                "description": "Percentage of exactly matching scores (rounded)",
                "range": [0.0, 1.0],
                "higher_is_better": True
            }
        ],
        "supported_sorting": ["quadratic_weighted_kappa", "pearson_correlation", "mean_absolute_error"]
    }