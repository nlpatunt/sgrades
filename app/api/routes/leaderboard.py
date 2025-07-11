from fastapi import APIRouter, HTTPException
from typing import List, Dict, Any, Optional
from datetime import datetime

from app.api.models.evaluation import LeaderboardEntry, BenchmarkReport
from app.config.datasets import get_all_datasets

router = APIRouter(prefix="/leaderboard", tags=["leaderboard"])

# Mock data for demonstration (replace with real data later)
mock_leaderboard_data = [
    LeaderboardEntry(
        rank=1,
        model_id="model_demo1",
        model_name="EssayGrader-Pro",
        submitter_name="Research Team A",
        avg_quadratic_weighted_kappa=0.847,
        avg_pearson_correlation=0.823,
        avg_accuracy=0.892,
        datasets_completed=12,
        total_datasets=12,
        submission_time=datetime(2025, 1, 10, 14, 30)
    ),
    LeaderboardEntry(
        rank=2,
        model_id="model_demo2", 
        model_name="AutoGrade-X",
        submitter_name="Research Team B",
        avg_quadratic_weighted_kappa=0.798,
        avg_pearson_correlation=0.775,
        avg_accuracy=0.856,
        datasets_completed=12,
        total_datasets=12,
        submission_time=datetime(2025, 1, 9, 16, 45)
    ),
    LeaderboardEntry(
        rank=3,
        model_id="model_demo3",
        model_name="GradeBot-2024",
        submitter_name="University Lab C",
        avg_quadratic_weighted_kappa=0.742,
        avg_pearson_correlation=0.719,
        avg_accuracy=0.801,
        datasets_completed=10,
        total_datasets=12,
        submission_time=datetime(2025, 1, 8, 9, 15)
    )
]

@router.get("/", response_model=List[LeaderboardEntry])
async def get_leaderboard(
    dataset_name: Optional[str] = None,
    metric: str = "avg_quadratic_weighted_kappa",
    limit: int = 50
):
    """Get the main leaderboard ranked by specified metric"""
    
    # Validate metric
    valid_metrics = ["avg_quadratic_weighted_kappa", "avg_pearson_correlation", "avg_accuracy"]
    if metric not in valid_metrics:
        raise HTTPException(
            status_code=400, 
            detail=f"Invalid metric. Choose from: {valid_metrics}"
        )
    
    # If dataset_name specified, filter for that dataset
    if dataset_name:
        # Validate dataset exists
        datasets = get_all_datasets()
        if dataset_name not in datasets:
            raise HTTPException(status_code=404, detail="Dataset not found")
        
        # TODO: Filter results for specific dataset
        # For now, return main leaderboard
        pass
    
    # Sort by specified metric and apply limit
    sorted_data = sorted(
        mock_leaderboard_data, 
        key=lambda x: getattr(x, metric, 0), 
        reverse=True
    )[:limit]
    
    # Update ranks based on sorting
    for i, entry in enumerate(sorted_data):
        entry.rank = i + 1
    
    return sorted_data

@router.get("/datasets/{dataset_name}")
async def get_dataset_leaderboard(dataset_name: str):
    """Get leaderboard for a specific dataset"""
    
    # Validate dataset exists
    datasets = get_all_datasets()
    if dataset_name not in datasets:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    # TODO: Return dataset-specific results
    # For now, return main leaderboard with dataset info
    return {
        "dataset_name": dataset_name,
        "dataset_description": datasets[dataset_name].description,
        "evaluation_metrics": datasets[dataset_name].evaluation_metrics,
        "leaderboard": mock_leaderboard_data,
        "total_models": len(mock_leaderboard_data)
    }

@router.get("/report", response_model=BenchmarkReport)
async def get_benchmark_report():
    """Get comprehensive benchmark report"""
    
    datasets = get_all_datasets()
    
    return BenchmarkReport(
        report_id=f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        generated_time=datetime.now(),
        total_models=len(mock_leaderboard_data),
        total_datasets=len(datasets),
        total_evaluations=len(mock_leaderboard_data) * len(datasets),
        leaderboard=mock_leaderboard_data,
        dataset_leaderboards={
            dataset_name: mock_leaderboard_data 
            for dataset_name in datasets.keys()
        },
        metric_distributions={
            "quadratic_weighted_kappa": {"mean": 0.796, "std": 0.053, "min": 0.742, "max": 0.847},
            "pearson_correlation": {"mean": 0.772, "std": 0.052, "min": 0.719, "max": 0.823},
            "accuracy": {"mean": 0.850, "std": 0.046, "min": 0.801, "max": 0.892}
        }
    )

@router.get("/stats")
async def get_leaderboard_stats():
    """Get general statistics about the benchmark"""
    
    datasets = get_all_datasets()
    
    return {
        "total_models_submitted": len(mock_leaderboard_data),
        "total_datasets": len(datasets),
        "total_evaluations_completed": len(mock_leaderboard_data) * len(datasets),
        "datasets_available": list(datasets.keys()),
        "latest_submission": max(entry.submission_time for entry in mock_leaderboard_data),
        "top_performer": mock_leaderboard_data[0].model_name if mock_leaderboard_data else None
    }