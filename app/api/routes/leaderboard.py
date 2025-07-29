from fastapi import APIRouter, HTTPException
from typing import List, Dict, Any, Optional
from datetime import datetime
import statistics

from app.services.database_service import DatabaseService

router = APIRouter(prefix="/leaderboard", tags=["leaderboard"])

@router.get("/")
async def get_leaderboard(metric: str = "quadratic_weighted_kappa", limit: int = 20):
    """Get leaderboard rankings from database"""
    
    try:
        print(f"🏆 Fetching leaderboard with metric: {metric}, limit: {limit}")
        
        # Get leaderboard from database
        leaderboard = DatabaseService.get_leaderboard(metric=metric, limit=limit)
        
        if not leaderboard:
            print("📋 No models found in database - returning empty leaderboard")
            return []
        
        print(f"📊 Retrieved {len(leaderboard)} models from database")
        
        # Log top performer for debugging
        if leaderboard:
            top_model = leaderboard[0]
            print(f"🥇 Top performer: {top_model['model_name']} (QWK: {top_model['avg_quadratic_weighted_kappa']})")
        
        return leaderboard
        
    except Exception as e:
        print(f"❌ Error getting leaderboard from database: {e}")
        import traceback
        traceback.print_exc()
        return []

@router.get("/datasets/{dataset_name}")
async def get_dataset_leaderboard(dataset_name: str):
    """Get leaderboard for a specific dataset"""
    
    try:
        print(f"📊 Fetching leaderboard for dataset: {dataset_name}")
        
        # For now, get general leaderboard (TODO: implement dataset-specific filtering)
        general_leaderboard = DatabaseService.get_leaderboard(limit=50)
        
        # TODO: Filter results for specific dataset from evaluation_results table
        # This would require a new DatabaseService method
        
        return {
            "dataset_name": dataset_name,
            "dataset_description": f"Performance rankings for {dataset_name} dataset",
            "leaderboard": general_leaderboard,
            "total_models": len(general_leaderboard),
            "note": "Dataset-specific filtering coming soon - showing all models for now"
        }
        
    except Exception as e:
        print(f"❌ Error getting dataset leaderboard: {e}")
        return {
            "dataset_name": dataset_name,
            "dataset_description": f"Error loading {dataset_name} leaderboard",
            "leaderboard": [],
            "total_models": 0,
            "error": str(e)
        }

@router.get("/report")
async def get_benchmark_report():
    """Get comprehensive benchmark report from database"""
    
    try:
        print("📋 Generating comprehensive benchmark report...")
        
        # Get platform statistics
        stats = DatabaseService.get_platform_stats()
        
        # Get full leaderboard for analysis
        leaderboard = DatabaseService.get_leaderboard(limit=100)
        
        # Calculate metric distributions
        metric_distributions = {}
        
        if leaderboard:
            # Extract metric values
            qwk_values = [entry['avg_quadratic_weighted_kappa'] for entry in leaderboard if entry['avg_quadratic_weighted_kappa'] is not None]
            pearson_values = [entry['avg_pearson_correlation'] for entry in leaderboard if entry['avg_pearson_correlation'] is not None]
            mae_values = [entry.get('avg_mean_absolute_error', 0) for entry in leaderboard if entry.get('avg_mean_absolute_error') is not None]
            f1_values = [entry.get('avg_f1_score', 0) for entry in leaderboard if entry.get('avg_f1_score') is not None]
            
            # Calculate distributions for each metric
            if qwk_values:
                metric_distributions["quadratic_weighted_kappa"] = {
                    "mean": round(statistics.mean(qwk_values), 3),
                    "median": round(statistics.median(qwk_values), 3),
                    "std": round(statistics.stdev(qwk_values) if len(qwk_values) > 1 else 0.0, 3),
                    "min": round(min(qwk_values), 3),
                    "max": round(max(qwk_values), 3),
                    "count": len(qwk_values)
                }
            
            if pearson_values:
                metric_distributions["pearson_correlation"] = {
                    "mean": round(statistics.mean(pearson_values), 3),
                    "median": round(statistics.median(pearson_values), 3),
                    "std": round(statistics.stdev(pearson_values) if len(pearson_values) > 1 else 0.0, 3),
                    "min": round(min(pearson_values), 3),
                    "max": round(max(pearson_values), 3),
                    "count": len(pearson_values)
                }
            
            if mae_values:
                metric_distributions["mean_absolute_error"] = {
                    "mean": round(statistics.mean(mae_values), 3),
                    "median": round(statistics.median(mae_values), 3),
                    "std": round(statistics.stdev(mae_values) if len(mae_values) > 1 else 0.0, 3),
                    "min": round(min(mae_values), 3),
                    "max": round(max(mae_values), 3),
                    "count": len(mae_values)
                }
            
            if f1_values:
                metric_distributions["f1_score"] = {
                    "mean": round(statistics.mean(f1_values), 3),
                    "median": round(statistics.median(f1_values), 3),
                    "std": round(statistics.stdev(f1_values) if len(f1_values) > 1 else 0.0, 3),
                    "min": round(min(f1_values), 3),
                    "max": round(max(f1_values), 3),
                    "count": len(f1_values)
                }
        
        # Performance tiers analysis
        performance_tiers = {
            "excellent": len([e for e in leaderboard if e['avg_quadratic_weighted_kappa'] >= 0.8]),
            "good": len([e for e in leaderboard if 0.6 <= e['avg_quadratic_weighted_kappa'] < 0.8]),
            "fair": len([e for e in leaderboard if 0.4 <= e['avg_quadratic_weighted_kappa'] < 0.6]),
            "poor": len([e for e in leaderboard if e['avg_quadratic_weighted_kappa'] < 0.4])
        }
        
        report = {
            "report_id": f"besesr_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "generated_time": datetime.now().isoformat(),
            "platform_version": "1.0.0",
            
            # Summary statistics
            "summary": {
                "total_models_submitted": stats['total_models_submitted'],
                "total_models_evaluated": len(leaderboard),
                "total_datasets": stats['total_datasets'],
                "total_evaluations_completed": stats['total_evaluations_completed'],
                "total_essays_evaluated": stats.get('total_essays_evaluated', 0),
                "latest_submission": stats.get('latest_submission'),
                "report_scope": "All evaluated models across all datasets"
            },
            
            # Top performers
            "top_performers": leaderboard[:5] if leaderboard else [],
            
            # Full leaderboard
            "leaderboard": leaderboard,
            
            # Statistical distributions
            "metric_distributions": metric_distributions,
            
            # Performance analysis
            "performance_analysis": {
                "performance_tiers": performance_tiers,
                "average_completion_rate": round(
                    statistics.mean([e['datasets_completed'] / e['total_datasets'] for e in leaderboard]) * 100, 1
                ) if leaderboard else 0,
                "most_challenging_datasets": [
                    "automatic_short_answer_grading_mohlar",
                    "SciEntSBank", 
                    "BEEtlE"
                ],  # TODO: Calculate from actual data
                "easiest_datasets": [
                    "ASAP-AES",
                    "persuade_2"
                ]  # TODO: Calculate from actual data
            },
            
            # Dataset information
            "datasets_info": {
                "total_datasets": 12,
                "dataset_categories": {
                    "essay_scoring": ["ASAP-AES", "ASAP2", "ASAP_plus_plus"],
                    "short_answer": ["ASAP-SAS", "SciEntSBank", "BEEtlE"],
                    "domain_specific": ["rice_chem", "CSEE", "EFL"],
                    "specialized": ["grade_like_a_human_dataset_os", "persuade_2", "automatic_short_answer_grading_mohlar"]
                }
            }
        }
        
        print(f"✅ Generated comprehensive report with {len(leaderboard)} models")
        return report
        
    except Exception as e:
        print(f"❌ Error generating benchmark report: {e}")
        import traceback
        traceback.print_exc()
        
        return {
            "report_id": "error_report",
            "generated_time": datetime.now().isoformat(),
            "error": str(e),
            "summary": {
                "total_models_submitted": 0,
                "total_models_evaluated": 0,
                "total_datasets": 12,
                "total_evaluations_completed": 0
            },
            "leaderboard": [],
            "metric_distributions": {}
        }

@router.get("/stats")
async def get_leaderboard_stats():
    """Get platform statistics from database"""
    
    try:
        print("📈 Fetching platform statistics from database...")
        
        # Get comprehensive stats from database
        stats = DatabaseService.get_platform_stats()
        
        # Get additional derived statistics
        leaderboard = DatabaseService.get_leaderboard(limit=100)
        
        # Find top performer
        top_performer = None
        if leaderboard:
            top_performer = leaderboard[0]['model_name']
        
        # Calculate additional metrics
        models_in_progress = 0  # TODO: Get from database
        avg_evaluation_time = "5-8 minutes"  # TODO: Calculate from actual data
        
        enhanced_stats = {
            # Core statistics
            "total_models_submitted": stats["total_models_submitted"],
            "total_models_evaluated": len(leaderboard),
            "total_datasets": stats["total_datasets"],
            "total_evaluations_completed": stats["total_evaluations_completed"],
            "total_essays_evaluated": stats.get("total_essays_evaluated", 0),
            
            # Performance statistics
            "top_performer": top_performer,
            "latest_submission": stats.get("latest_submission"),
            "models_in_progress": models_in_progress,
            "avg_evaluation_time": avg_evaluation_time,
            
            # Dataset information
            "datasets_available": [
                "ASAP-AES", "ASAP-SAS", "rice_chem", "CSEE", "EFL", 
                "grade_like_a_human_dataset_os", "persuade_2", "ASAP2", 
                "ASAP_plus_plus", "SciEntSBank", "BEEtlE", 
                "automatic_short_answer_grading_mohlar"
            ],
            
            # Platform metrics
            "platform_uptime_days": stats.get("platform_uptime_days", 1),
            "database_status": "connected",
            "api_version": "1.0.0",
            
            # Success rates (TODO: calculate from actual data)
            "evaluation_success_rate": 0.95,
            "avg_datasets_per_model": round(
                statistics.mean([e['datasets_completed'] for e in leaderboard]), 1
            ) if leaderboard else 0
        }
        
        print(f"📊 Platform stats: {enhanced_stats['total_models_submitted']} models, {enhanced_stats['total_evaluations_completed']} evaluations")
        
        return enhanced_stats
        
    except Exception as e:
        print(f"❌ Error getting platform statistics: {e}")
        import traceback
        traceback.print_exc()
        
        # Return fallback stats
        return {
            "total_models_submitted": 0,
            "total_models_evaluated": 0,
            "total_datasets": 12,
            "total_evaluations_completed": 0,
            "total_essays_evaluated": 0,
            "datasets_available": [],
            "latest_submission": None,
            "top_performer": None,
            "database_status": "error",
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

@router.get("/health")
async def leaderboard_health():
    """Health check for leaderboard service"""
    
    try:
        # Test database connection
        stats = DatabaseService.get_platform_stats()
        leaderboard_count = len(DatabaseService.get_leaderboard(limit=1))
        
        return {
            "status": "healthy",
            "service": "leaderboard",
            "database_connection": "ok",
            "models_available": stats["total_models_submitted"],
            "evaluations_completed": stats["total_evaluations_completed"],
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