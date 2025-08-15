# app/api/routes/output_submissions.py

from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from typing import List, Dict, Any, Optional
import pandas as pd
import io
import os
import zipfile
import tempfile
from datetime import datetime
import statistics

from app.services.database_service import DatabaseService
from app.utils.metrics import calculate_evaluation_metrics
from app.services.dataset_loader import BESESRDatasetManager
from app.models.pydantic_models import (
    BenchmarkSubmissionResponse, SingleTestResponse, CSVValidationResponse,
    SubmissionTemplate, SubmissionStatus, RecentSubmissions, SubmissionsHealthCheck,
    LeaderboardResponse, ResearcherProgress, EvaluationMetrics
)

# Initialize dataset manager
dataset_manager = BESESRDatasetManager()
router = APIRouter(prefix="/submissions", tags=["submissions"])

class ResultsSubmission:
    """Handle CSV results submission and validation"""
    
    @staticmethod
    def validate_results_csv(df: pd.DataFrame, dataset_name: str) -> Dict[str, Any]:
        """Validate uploaded results CSV"""
        
        required_columns = ['essay_id', 'predicted_score']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            return {
                'valid': False,
                'error': f"Missing required columns: {missing_columns}",
                'required_columns': required_columns
            }
        
        # Check for valid scores
        try:
            df['predicted_score'] = pd.to_numeric(df['predicted_score'])
        except:
            return {
                'valid': False,
                'error': "predicted_score column must contain numeric values"
            }
        
        # Check for duplicates
        duplicates = df['essay_id'].duplicated().sum()
        if duplicates > 0:
            return {
                'valid': False,
                'error': f"Found {duplicates} duplicate essay_ids"
            }
        
        return {
            'valid': True,
            'essay_count': len(df),
            'score_range': (df['predicted_score'].min(), df['predicted_score'].max())
        }

@router.post("/upload-complete-benchmark", response_model=BenchmarkSubmissionResponse)
async def upload_complete_benchmark(
    submitter_name: str = Form(...),
    submitter_email: str = Form(...),
    model_name: str = Form(...),
    model_description: Optional[str] = Form(None),
    csv_files: List[UploadFile] = File(...)
):
    """Upload complete benchmark with all 15 datasets"""
    
    try:
        print(f"🚀 Processing complete benchmark submission from {submitter_name}")
        print(f"   Model: {model_name}")
        print(f"   Files received: {len(csv_files)}")
        
        # Validate we have exactly 15 files
        if len(csv_files) != 15:
            raise HTTPException(
                status_code=400, 
                detail=f"Complete benchmark requires exactly 15 CSV files. Received {len(csv_files)} files."
            )
        
        # Validate all files are CSV
        for file in csv_files:
            if not file.filename.endswith('.csv'):
                raise HTTPException(
                    status_code=400, 
                    detail=f"All files must be CSV format. Invalid file: {file.filename}"
                )
        
        # Create uploads directory
        uploads_dir = "uploads/complete_benchmarks"
        os.makedirs(uploads_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        benchmark_dir = os.path.join(uploads_dir, f"{submitter_name}_{timestamp}")
        os.makedirs(benchmark_dir, exist_ok=True)
        
        # Process each dataset
        dataset_results = {}
        total_essays = 0
        successful_datasets = []
        failed_datasets = []
        
        for file in csv_files:
            try:
                # Extract dataset name from filename
                dataset_name = file.filename.replace('.csv', '').replace('_results', '')
                
                print(f"   📊 Processing {dataset_name}...")
                
                # Validate dataset exists
                if dataset_name not in dataset_manager.datasets_config:
                    available_datasets = list(dataset_manager.datasets_config.keys())
                    print(f"   ⚠️ Unknown dataset: {dataset_name}. Available: {available_datasets}")
                    failed_datasets.append(dataset_name)
                    continue
                
                # Read and validate CSV
                content = await file.read()
                df = pd.read_csv(io.StringIO(content.decode('utf-8')))
                
                # Validate CSV format
                validation = ResultsSubmission.validate_results_csv(df, dataset_name)
                if not validation['valid']:
                    print(f"   ❌ Invalid CSV for {dataset_name}: {validation['error']}")
                    failed_datasets.append(dataset_name)
                    continue
                
                # Save file
                file_path = os.path.join(benchmark_dir, f"{dataset_name}.csv")
                with open(file_path, 'wb') as f:
                    f.write(content)
                
                # Load reference dataset
                original_essays = dataset_manager.load_dataset_for_evaluation(dataset_name, sample_size=1000)
                
                if not original_essays:
                    print(f"   ❌ Could not load reference dataset: {dataset_name}")
                    failed_datasets.append(dataset_name)
                    continue
                
                # Match predictions with human scores
                human_scores_lookup = {essay['essay_id']: essay['human_score'] for essay in original_essays}
                matched_data = []
                
                for _, row in df.iterrows():
                    essay_id = str(row['essay_id'])
                    predicted_score = row['predicted_score']
                    
                    if essay_id in human_scores_lookup:
                        matched_data.append({
                            'essay_id': essay_id,
                            'predicted_score': float(predicted_score),
                            'human_score': human_scores_lookup[essay_id]
                        })
                    else:
                        # Try partial matching
                        for essay in original_essays:
                            if essay_id in str(essay['essay_id']) or str(essay['essay_id']) in essay_id:
                                matched_data.append({
                                    'essay_id': essay_id,
                                    'predicted_score': float(predicted_score),
                                    'human_score': essay['human_score']
                                })
                                break
                
                if not matched_data:
                    print(f"   ❌ No matching essays found for {dataset_name}")
                    failed_datasets.append(dataset_name)
                    continue
                
                # Calculate metrics
                metrics = calculate_evaluation_metrics(matched_data)
                
                dataset_results[dataset_name] = {
                    'quadratic_weighted_kappa': metrics.get('quadratic_weighted_kappa', 0.0),
                    'pearson_correlation': metrics.get('pearson_correlation', 0.0),
                    'spearman_correlation': metrics.get('spearman_correlation', 0.0),
                    'mean_absolute_error': metrics.get('mean_absolute_error', 999.0),
                    'root_mean_squared_error': metrics.get('root_mean_squared_error', 999.0),
                    'f1_score': metrics.get('f1_score', 0.0),
                    'accuracy': metrics.get('accuracy', 0.0),
                    'essays_evaluated': len(matched_data),
                    'match_rate': len(matched_data) / len(df) if len(df) > 0 else 0
                }
                
                total_essays += len(matched_data)
                successful_datasets.append(dataset_name)
                
                print(f"   ✅ {dataset_name}: QWK={metrics.get('quadratic_weighted_kappa', 0):.3f}, Essays={len(matched_data)}")
                
            except Exception as e:
                print(f"   ❌ Error processing {dataset_name}: {e}")
                failed_datasets.append(dataset_name)
                continue
        
        # Check if we have enough successful datasets
        if len(successful_datasets) < 10:  # Require at least 10/15 for now
            raise HTTPException(
                status_code=400,
                detail=f"Insufficient successful datasets. Got {len(successful_datasets)}, need at least 10. Failed: {failed_datasets}"
            )
        
        # Calculate aggregate metrics
        if dataset_results:
            avg_qwk = statistics.mean([r['quadratic_weighted_kappa'] for r in dataset_results.values()])
            avg_pearson = statistics.mean([r['pearson_correlation'] for r in dataset_results.values()])
            avg_mae = statistics.mean([r['mean_absolute_error'] for r in dataset_results.values()])
            avg_f1 = statistics.mean([r['f1_score'] for r in dataset_results.values()])
            avg_accuracy = statistics.mean([r['accuracy'] for r in dataset_results.values()])
        else:
            raise HTTPException(status_code=400, detail="No datasets were successfully processed")
        
        # Save individual submissions to database
        submission_ids = []
        for dataset_name, metrics in dataset_results.items():
            submission_data = {
                "dataset_name": dataset_name,
                "submitter_name": submitter_name,
                "submitter_email": submitter_email,
                "file_path": os.path.join(benchmark_dir, f"{dataset_name}.csv"),
                "file_format": "csv",
                "status": "completed",
                "description": f"{model_name}: {dataset_name} - Complete Benchmark"
            }
            
            submission_dict = DatabaseService.save_output_submission(submission_data)
            submission_id = submission_dict["id"]
            
            # Update with evaluation results
            DatabaseService.update_output_submission_status(
                submission_id, "completed", metrics, processing_time=1.0
            )
            
            # Save detailed evaluation result
            evaluation_data = {
                "submission_id": submission_id,
                "dataset_name": dataset_name,
                "metrics": metrics,
                "essays_evaluated": metrics['essays_evaluated'],
                "match_rate": metrics['match_rate']
            }
            
            DatabaseService.save_evaluation_result(evaluation_data)
            submission_ids.append(submission_id)
        
        print(f"   🎉 Complete benchmark processed!")
        print(f"   📊 Successful datasets: {len(successful_datasets)}")
        print(f"   📊 Average QWK: {avg_qwk:.3f}")
        print(f"   📊 Total essays: {total_essays}")
        
        return BenchmarkSubmissionResponse(
            message='Complete benchmark processed successfully!',
            submitter_name=submitter_name,
            model_name=model_name,
            datasets_processed=len(successful_datasets),
            failed_datasets=failed_datasets,
            total_essays_evaluated=total_essays,
            submission_ids=submission_ids,
            avg_quadratic_weighted_kappa=avg_qwk,
            avg_pearson_correlation=avg_pearson,
            avg_mean_absolute_error=avg_mae,
            avg_f1_score=avg_f1,
            avg_accuracy=avg_accuracy,
            status='completed',
            benchmark_type='complete_15_dataset',
            completion_rate=(len(successful_datasets) / 15) * 100
        )
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error processing complete benchmark: {e}")
        raise HTTPException(status_code=500, detail=f"Benchmark processing error: {str(e)}")

@router.post("/upload-single-result", response_model=SingleTestResponse)
async def upload_single_result(
    model_name: str = Form(...),
    dataset_name: str = Form(...),
    submitter_name: str = Form(...),
    submitter_email: str = Form(...),
    result_file: UploadFile = File(...),
    model_description: Optional[str] = Form(None)
):
    """Upload CSV result for a single dataset (testing only - NOT for leaderboard)"""
    
    try:
        print(f"🧪 Processing individual test submission: {dataset_name} from {submitter_name}")
        print(f"   ⚠️ NOTE: This is for testing only and will NOT appear on the main leaderboard")
        
        # Validate file type
        if not result_file.filename.endswith('.csv'):
            raise HTTPException(status_code=400, detail="File must be a CSV format")
        
        # Check if dataset exists
        if dataset_name not in dataset_manager.datasets_config:
            available_datasets = list(dataset_manager.datasets_config.keys())
            raise HTTPException(
                status_code=400, 
                detail=f"Unknown dataset: {dataset_name}. Available datasets: {available_datasets}"
            )
        
        # Read and validate CSV
        content = await result_file.read()
        df = pd.read_csv(io.StringIO(content.decode('utf-8')))
        
        print(f"   📊 CSV loaded: {df.shape[0]} rows, {df.shape[1]} columns")
        
        # Validate CSV format
        validation = ResultsSubmission.validate_results_csv(df, dataset_name)
        if not validation['valid']:
            raise HTTPException(status_code=400, detail=f"Invalid CSV: {validation['error']}")
        
        # Save file to uploads directory
        uploads_dir = "uploads/individual_tests"
        os.makedirs(uploads_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_path = os.path.join(uploads_dir, f"{dataset_name}_{timestamp}.csv")
        
        with open(file_path, 'wb') as f:
            f.write(content)
        
        # Save submission to database (marked as individual test)
        submission_data = {
            "dataset_name": dataset_name,
            "submitter_name": f"TEST_{submitter_name}",  # Mark as test
            "submitter_email": submitter_email,
            "file_path": file_path,
            "file_format": "csv",
            "status": "processing",
            "description": f"[TEST] {model_name}: {dataset_name} - Individual Test"
        }
        
        submission_dict = DatabaseService.save_output_submission(submission_data)
        submission_id = submission_dict["id"]
        
        # Load original dataset for evaluation
        original_essays = dataset_manager.load_dataset_for_evaluation(dataset_name, sample_size=1000)
        
        if not original_essays:
            DatabaseService.update_output_submission_status(
                submission_id, "failed", error_message="Could not load reference dataset"
            )
            raise HTTPException(status_code=500, detail=f"Could not load reference dataset: {dataset_name}")
        
        # Match predictions with human scores
        human_scores_lookup = {essay['essay_id']: essay['human_score'] for essay in original_essays}
        matched_data = []
        unmatched_count = 0
        
        for _, row in df.iterrows():
            essay_id = str(row['essay_id'])
            predicted_score = row['predicted_score']
            
            if essay_id in human_scores_lookup:
                matched_data.append({
                    'essay_id': essay_id,
                    'predicted_score': float(predicted_score),
                    'human_score': human_scores_lookup[essay_id]
                })
            else:
                # Try partial matching
                for essay in original_essays:
                    if essay_id in str(essay['essay_id']) or str(essay['essay_id']) in essay_id:
                        matched_data.append({
                            'essay_id': essay_id,
                            'predicted_score': float(predicted_score),
                            'human_score': essay['human_score']
                        })
                        break
                else:
                    unmatched_count += 1
        
        if not matched_data:
            DatabaseService.update_output_submission_status(
                submission_id, "failed", error_message="No matching essays found"
            )
            raise HTTPException(status_code=400, detail="No matching essays found")
        
        # Calculate evaluation metrics
        metrics = calculate_evaluation_metrics(matched_data)
        
        evaluation_result = EvaluationMetrics(
            quadratic_weighted_kappa=metrics.get('quadratic_weighted_kappa', 0.0),
            pearson_correlation=metrics.get('pearson_correlation', 0.0),
            spearman_correlation=metrics.get('spearman_correlation', 0.0),
            mean_absolute_error=metrics.get('mean_absolute_error', 999.0),
            root_mean_squared_error=metrics.get('root_mean_squared_error', 999.0),
            f1_score=metrics.get('f1_score', 0.0),
            accuracy=metrics.get('accuracy', 0.0),
            essays_evaluated=len(matched_data),
            match_rate=len(matched_data) / len(df) if len(df) > 0 else 0
        )
        
        # Update submission with results
        DatabaseService.update_output_submission_status(
            submission_id, "completed", evaluation_result.dict(), processing_time=1.0
        )
        
        print(f"   ✅ Individual test completed: QWK={evaluation_result.quadratic_weighted_kappa:.3f}")
        
        return SingleTestResponse(
            message='Individual test completed successfully!',
            note='This is a test submission and does NOT appear on the main leaderboard.',
            submission_id=submission_id,
            model_name=model_name,
            dataset_name=dataset_name,
            evaluation_results=evaluation_result,
            status='completed',
            submission_type='individual_test'
        )
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error processing individual test: {e}")
        raise HTTPException(status_code=500, detail=f"Test processing error: {str(e)}")

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

@router.get("/leaderboard", response_model=LeaderboardResponse)
async def get_submissions_leaderboard(
    dataset_name: Optional[str] = None,
    metric: str = "quadratic_weighted_kappa",
    limit: int = 10,
    type: str = "complete"  # "complete" or "individual"
):
    """Get leaderboard rankings"""
    try:
        if type == "complete":
            leaderboard_entries = DatabaseService.get_complete_benchmark_leaderboard(
                metric=f"avg_{metric}",
                limit=limit
            )
        else:
            leaderboard_entries = DatabaseService.get_output_leaderboard(limit=limit)
        
        return LeaderboardResponse(
            leaderboard_type=type,
            metric=metric,
            limit=limit,
            leaderboard=leaderboard_entries
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching leaderboard: {str(e)}")

@router.post("/validate-csv", response_model=CSVValidationResponse)
async def validate_csv_format(file: UploadFile = File(...)):
    """Validate CSV format before submission"""
    
    try:
        if not file.filename.endswith('.csv'):
            return CSVValidationResponse(
                valid=False,
                error="File must be a CSV",
                required_columns=['essay_id', 'predicted_score']
            )
        
        # Read CSV content
        content = await file.read()
        df = pd.read_csv(io.StringIO(content.decode('utf-8')))
        
        # Validate format
        validation = ResultsSubmission.validate_results_csv(df, "generic")
        
        if validation['valid']:
            return CSVValidationResponse(
                valid=True,
                message='CSV format is valid',
                row_count=len(df),
                columns=list(df.columns),
                sample_data=df.head(3).to_dict('records') if not df.empty else []
            )
        else:
            return CSVValidationResponse(
                valid=False,
                error=validation['error'],
                required_columns=validation.get('required_columns', []),
                columns_found=list(df.columns)
            )
            
    except Exception as e:
        return CSVValidationResponse(
            valid=False,
            error=f"File validation failed: {str(e)}"
        )

@router.get("/template", response_model=SubmissionTemplate)
async def get_submission_template():
    """Get CSV template and upload instructions"""
    
    # Get available datasets
    try:
        available_datasets = list(dataset_manager.datasets_config.keys())
    except:
        available_datasets = []
    
    template_data = SubmissionTemplate(
        csv_format={
            "required_columns": ["essay_id", "predicted_score"],
            "example_rows": [
                {"essay_id": "ASAP-AES_001", "predicted_score": 3.5},
                {"essay_id": "ASAP-AES_002", "predicted_score": 4.2},
                {"essay_id": "rice_chem_001", "predicted_score": 2.8}
            ]
        },
        available_datasets=available_datasets,
        upload_instructions={
            "complete_benchmark": {
                "step1": "Download all 15 datasets",
                "step2": "Train your model on train/validation splits", 
                "step3": "Generate predictions on test splits (unlabeled)",
                "step4": "Upload all 15 CSV files at once",
                "step5": "Automatic evaluation and leaderboard ranking"
            },
            "individual_testing": {
                "step1": "Select single dataset for testing",
                "step2": "Upload CSV with predictions", 
                "step3": "Get immediate feedback (not on leaderboard)",
                "step4": "Use for development and debugging"
            }
        },
        requirements={
            "complete_benchmark": {
                "files_required": 15,
                "file_formats_accepted": [".csv"],
                "max_file_size": "10MB per file",
                "required_columns": ["essay_id", "predicted_score"]
            },
            "individual_testing": {
                "files_required": 1,
                "file_formats_accepted": [".csv"],
                "max_file_size": "10MB",
                "required_columns": ["essay_id", "predicted_score"]
            }
        },
        example_csv="""essay_id,predicted_score
ASAP-AES_001,3.5
ASAP-AES_002,4.2
ASAP-AES_003,2.8"""
    )
    
    return template_data

@router.get("/submission-status/{submission_id}", response_model=SubmissionStatus)
async def get_submission_status(submission_id: int):
    """Get status of a submission"""
    
    try:
        submission = DatabaseService.get_output_submission(submission_id)
        if not submission:
            raise HTTPException(status_code=404, detail="Submission not found")
        
        # Get evaluation results if completed
        evaluation_results = DatabaseService.get_evaluation_results_by_submission(submission_id)
        
        return SubmissionStatus(
            submission_id=submission_id,
            dataset_name=submission['dataset_name'],
            submitter_name=submission['submitter_name'],
            status=submission['status'],
            submitted_at=submission['submission_time'],
            evaluation_result=submission.get('evaluation_result'),
            detailed_evaluations=evaluation_results,
            error_message=submission.get('error_message')
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving status: {str(e)}")

@router.get("/recent", response_model=RecentSubmissions)
async def get_recent_submissions(limit: int = 10, type: str = "all"):
    """Get recent submissions for monitoring"""
    
    try:
        submissions = DatabaseService.get_recent_submissions(limit)
        
        # Filter by type if specified
        if type == "complete":
            submissions = [s for s in submissions if not s['submitter_name'].startswith('TEST_')]
        elif type == "individual":
            submissions = [s for s in submissions if s['submitter_name'].startswith('TEST_')]
        
        return RecentSubmissions(
            submissions=submissions,
            total_count=len(submissions),
            type_filter=type
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving submissions: {str(e)}")

@router.get("/health", response_model=SubmissionsHealthCheck)
async def submissions_health():
    """Health check for submissions service"""
    return SubmissionsHealthCheck(
        service='submissions',
        status='healthy',
        timestamp=datetime.now().isoformat(),
        endpoints=[
            'POST /upload-complete-benchmark - Upload 15 datasets for leaderboard',
            'POST /upload-single-result - Upload single dataset for testing',
            'POST /validate-csv - Validate CSV format',
            'GET /template - Get upload instructions',
            'GET /progress/{name} - Check researcher progress'
        ],
        benchmark_requirements={
            'total_datasets': 15,
            'file_format': 'CSV',
            'required_columns': ['essay_id', 'predicted_score']
        }
    )