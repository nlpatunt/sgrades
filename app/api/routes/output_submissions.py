# app/api/routes/output_submissions.py (Fixed session management)

from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from typing import List, Dict, Any, Optional
import pandas as pd
import io
import uuid
from datetime import datetime
import os
import time
import zipfile
import tempfile
from pathlib import Path 

from app.services.database_service import DatabaseService
from app.utils.metrics import calculate_evaluation_metrics
from app.services.dataset_loader import dataset_manager

router = APIRouter(prefix="/submissions", tags=["submissions"])

class ResultsSubmission:
    """Handle CSV results submission and evaluation"""
    
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

# Just the key part of upload endpoint that needs to change:

@router.post("/upload-single-result")
async def upload_single_result(
    model_name: str = Form(...),
    dataset_name: str = Form(...),
    submitter_name: str = Form(...),
    submitter_email: str = Form(...),
    model_description: Optional[str] = Form(None),
    result_file: UploadFile = File(...)
):
    """Upload single CSV result for a specific dataset"""
    
    try:
        print(f"📤 Processing single file submission from {submitter_name}")
        print(f"   Model: {model_name}")
        print(f"   Dataset: {dataset_name}")
        print(f"   File: {result_file.filename}")
        start_time = time.time()
        
        # Basic file validation
        if not result_file.filename.endswith('.csv'):
            raise HTTPException(status_code=400, detail=f"File {result_file.filename} must be a CSV file")
        
        # Check if dataset exists
        if dataset_name not in dataset_manager.datasets_config:
            raise HTTPException(status_code=400, detail=f"Unknown dataset: {dataset_name}")
        
        # Read CSV content
        content = await result_file.read()
        df = pd.read_csv(io.StringIO(content.decode('utf-8')))
        
        print(f"   📊 CSV loaded: {df.shape[0]} rows, {df.shape[1]} columns")
        print(f"   📊 Columns: {list(df.columns)}")
        
        # Validate CSV format
        validation = ResultsSubmission.validate_results_csv(df, dataset_name)
        if not validation['valid']:
            raise HTTPException(status_code=400, detail=f"Invalid CSV: {validation['error']}")
        
        # Save submission to database and get the dictionary
        file_path = f"uploads/{dataset_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        submission_data = {
            "dataset_name": dataset_name,
            "submitter_name": submitter_name,
            "submitter_email": submitter_email,
            "file_path": file_path,
            "file_format": "csv",
            "status": "processing",
            "description": f"{model_name}: {model_description}" if model_description else model_name
        }
        
        # This now returns a dictionary, not an object
        submission_dict = DatabaseService.save_output_submission(submission_data)
        submission_id = submission_dict["id"]  # Get ID from dictionary
        print(f"   💾 Saved submission {submission_id}")
        
        # Load original dataset for evaluation
        print(f"   📖 Loading original dataset: {dataset_name}")
        original_essays = dataset_manager.load_dataset_for_evaluation(dataset_name, sample_size=1000)
        human_scores_lookup = {essay['essay_id']: essay['human_score'] for essay in original_essays}
        
        # Match predictions with human scores
        print(f"   🔍 Matching predictions with human scores...")
        matched_data = []
        for _, row in df.iterrows():
            essay_id = row['essay_id']
            predicted_score = row['predicted_score']
            
            if essay_id in human_scores_lookup:
                human_score = human_scores_lookup[essay_id]
                matched_data.append({
                    'essay_id': essay_id,
                    'human_score': human_score,
                    'predicted_score': predicted_score
                })
        
        if not matched_data:
            error_msg = "No matching essays found between submission and original dataset"
            DatabaseService.update_output_submission_status(
                submission_id, "failed", error_message=error_msg
            )
            raise HTTPException(status_code=400, detail=error_msg)
        
        print(f"   ✅ Matched {len(matched_data)} essays out of {len(df)} submitted")
        
        # Calculate metrics
        print(f"   🧮 Calculating evaluation metrics...")
        human_scores = [item['human_score'] for item in matched_data]
        predicted_scores = [item['predicted_score'] for item in matched_data]
        metrics = calculate_evaluation_metrics(human_scores, predicted_scores)
        
        # Create evaluation result
        evaluation_result = {
            'quadratic_weighted_kappa': metrics['qwk'],
            'pearson_correlation': metrics['pearson'],
            'spearman_correlation': metrics['spearman'],
            'mean_absolute_error': metrics['mae'],
            'root_mean_squared_error': metrics['rmse'],
            'f1_score': metrics['f1'],
            'accuracy': metrics['accuracy'],
            'essays_evaluated': len(matched_data),
            'detailed_metrics': {
                'matched_essays': len(matched_data),
                'submitted_essays': len(df),
                'match_rate': len(matched_data) / len(df),
                'model_name': model_name,
                'submission_method': 'single_csv_upload'
            }
        }
        
        # Update submission with results
        processing_time = time.time() - start_time
        DatabaseService.update_output_submission_status(
            submission_id, 
            "completed", 
            evaluation_result=evaluation_result,
            processing_time=processing_time
        )
        
        # Save detailed evaluation
        DatabaseService.save_evaluation_result(submission_id, dataset_name, evaluation_result)
        
        print(f"   🎉 Evaluation completed: QWK={metrics['qwk']:.3f}")
        
        return {
            'message': 'File processed and evaluated successfully!',
            'submission_id': submission_id,
            'model_name': model_name,
            'dataset_name': dataset_name,
            'evaluation_results': {
                'quadratic_weighted_kappa': metrics['qwk'],
                'pearson_correlation': metrics['pearson'],
                'mean_absolute_error': metrics['mae'],
                'accuracy': metrics['accuracy'],
                'essays_evaluated': len(matched_data),
                'match_rate': len(matched_data) / len(df)
            },
            'processing_time_seconds': processing_time,
            'leaderboard_url': '/leaderboard/'
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")

@router.post("/upload-results")
async def upload_results(
    model_name: str = Form(...),
    submitter_name: str = Form(...),
    submitter_email: str = Form(...),
    model_description: Optional[str] = Form(None),
    paper_url: Optional[str] = Form(None),
    results_files: List[UploadFile] = File(...)
):
    """Upload CSV results from model evaluation on downloaded datasets"""
    
    try:
        print(f"📤 Processing multiple file submission from {submitter_name}")
        start_time = time.time()
        
        # Process uploaded files
        dataset_results = {}
        
        for file in results_files:
            if not file.filename.endswith('.csv'):
                raise HTTPException(status_code=400, detail=f"File {file.filename} must be a CSV file")
            
            # Extract dataset name from filename
            dataset_name = file.filename.replace('_results.csv', '').replace('.csv', '')
            
            if dataset_name not in dataset_manager.datasets_config:
                raise HTTPException(status_code=400, detail=f"Unknown dataset: {dataset_name}")
            
            # Read CSV content
            content = await file.read()
            df = pd.read_csv(io.StringIO(content.decode('utf-8')))
            
            # Validate CSV format
            validation = ResultsSubmission.validate_results_csv(df, dataset_name)
            if not validation['valid']:
                raise HTTPException(status_code=400, detail=f"Invalid CSV for {dataset_name}: {validation['error']}")
            
            dataset_results[dataset_name] = {
                'dataframe': df,
                'validation': validation,
                'filename': file.filename
            }
            
            print(f"   ✅ Validated {dataset_name}: {validation['essay_count']} predictions")
        
        if not dataset_results:
            raise HTTPException(status_code=400, detail="No valid dataset results found")
        
        # Process each dataset and save individual submissions
        submission_results = []
        total_metrics = []
        
        for dataset_name, results in dataset_results.items():
            try:
                print(f"🧮 Processing {dataset_name}...")
                
                # Create file path
                file_path = f"uploads/{dataset_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                
                # Save submission to database first and get ID immediately
                submission_data = {
                    "dataset_name": dataset_name,
                    "submitter_name": submitter_name,
                    "submitter_email": submitter_email,
                    "file_path": file_path,
                    "file_format": "csv",
                    "status": "processing",
                    "description": f"{model_name}: {model_description}" if model_description else model_name
                }
                
                submission_dict = DatabaseService.save_output_submission(submission_data)
                submission_id = submission_dict["id"]
                
                # Load original dataset to get human scores
                original_essays = dataset_manager.load_dataset_for_evaluation(dataset_name, sample_size=1000)
                
                # Create lookup for human scores
                human_scores_lookup = {essay['essay_id']: essay['human_score'] for essay in original_essays}
                
                # Match predictions with human scores
                results_df = results['dataframe']
                matched_data = []
                
                for _, row in results_df.iterrows():
                    essay_id = row['essay_id']
                    predicted_score = row['predicted_score']
                    
                    if essay_id in human_scores_lookup:
                        human_score = human_scores_lookup[essay_id]
                        matched_data.append({
                            'essay_id': essay_id,
                            'human_score': human_score,
                            'predicted_score': predicted_score
                        })
                
                if not matched_data:
                    error_msg = f"No matching essays found between submission and original dataset"
                    DatabaseService.update_output_submission_status(
                        submission_id, "failed", error_message=error_msg
                    )
                    print(f"   ❌ {dataset_name}: {error_msg}")
                    continue
                
                # Extract scores for metrics calculation
                human_scores = [item['human_score'] for item in matched_data]
                predicted_scores = [item['predicted_score'] for item in matched_data]
                
                # Calculate metrics
                metrics = calculate_evaluation_metrics(human_scores, predicted_scores)
                
                # Create evaluation result
                evaluation_result = {
                    'quadratic_weighted_kappa': metrics['qwk'],
                    'pearson_correlation': metrics['pearson'],
                    'spearman_correlation': metrics['spearman'],
                    'mean_absolute_error': metrics['mae'],
                    'root_mean_squared_error': metrics['rmse'],
                    'f1_score': metrics['f1'],
                    'accuracy': metrics['accuracy'],
                    'essays_evaluated': len(matched_data),
                    'evaluation_duration': 0.0,
                    'status': 'completed',
                    'detailed_metrics': {
                        'matched_essays': len(matched_data),
                        'submitted_essays': len(results_df),
                        'match_rate': len(matched_data) / len(results_df),
                        'score_distribution': metrics.get('distribution', {}),
                        'submission_method': 'csv_upload',
                        'model_name': model_name
                    }
                }
                
                # Update submission with results
                processing_time = time.time() - start_time
                DatabaseService.update_output_submission_status(
                    submission_id, 
                    "completed", 
                    evaluation_result=evaluation_result,
                    processing_time=processing_time
                )
                
                # Also save detailed evaluation result
                DatabaseService.save_evaluation_result(submission_id, dataset_name, evaluation_result)
                
                submission_results.append({
                    'submission_id': submission_id,
                    'dataset_name': dataset_name,
                    'metrics': metrics,
                    'essays_evaluated': len(matched_data)
                })
                
                total_metrics.append(metrics)
                
                print(f"   ✅ {dataset_name}: QWK={metrics['qwk']:.3f}, Pearson={metrics['pearson']:.3f}, Essays={len(matched_data)}")
                
            except Exception as e:
                print(f"   ❌ Error processing {dataset_name}: {e}")
                # Update submission with error (using submission_id we stored earlier)
                if 'submission_id' in locals():
                    DatabaseService.update_output_submission_status(
                        submission_id, "failed", error_message=str(e)
                    )
        
        # Calculate overall results
        total_processing_time = time.time() - start_time
        
        if total_metrics:
            avg_qwk = sum(m['qwk'] for m in total_metrics) / len(total_metrics)
            print(f"🎉 Submission completed! Average QWK: {avg_qwk:.3f}")
            
            return {
                'message': 'Results submitted and evaluated successfully!',
                'submitter_name': submitter_name,
                'model_name': model_name,
                'datasets_processed': len(dataset_results),
                'successful_evaluations': len(total_metrics),
                'failed_evaluations': len(dataset_results) - len(total_metrics),
                'average_qwk': avg_qwk,
                'average_pearson': sum(m['pearson'] for m in total_metrics) / len(total_metrics),
                'total_essays_evaluated': sum(r['essays_evaluated'] for r in submission_results),
                'processing_time_seconds': total_processing_time,
                'submission_results': submission_results,
                'leaderboard_url': '/leaderboard/'
            }
        else:
            return {
                'message': 'Submission processed but no evaluations completed successfully',
                'submitter_name': submitter_name,
                'model_name': model_name,
                'datasets_processed': len(dataset_results),
                'successful_evaluations': 0,
                'failed_evaluations': len(dataset_results),
                'errors': 'Check individual dataset processing logs'
            }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error processing submission: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to process submission: {str(e)}")

@router.get("/template")
async def get_submission_template():
    """Get template showing expected CSV format"""
    
    example_csv = """essay_id,predicted_score
ASAP-AES_0,3.5
ASAP-AES_1,4.2
ASAP-AES_2,2.8
rice_chem_0,4.0
rice_chem_1,3.3
"""
    
    return {
        'description': 'Expected CSV format for results submission',
        'required_columns': ['essay_id', 'predicted_score'],
        'example_csv': example_csv,
        'instructions': [
            '1. Download datasets using /datasets/download/all',
            '2. Run your model on each dataset',
            '3. Create one CSV file per dataset with essay_id and predicted_score columns',
            '4. Upload CSV files using the submission form',
            '5. Platform will calculate metrics and update leaderboard'
        ]
    }

@router.post("/validate-csv")
async def validate_csv_format(
    dataset_name: str = Form(...),
    csv_file: UploadFile = File(...)
):
    """Validate CSV format before submission"""
    
    try:
        if not csv_file.filename.endswith('.csv'):
            raise HTTPException(status_code=400, detail="File must be a CSV")
        
        # Read CSV content
        content = await csv_file.read()
        df = pd.read_csv(io.StringIO(content.decode('utf-8')))
        
        # Validate format
        validation = ResultsSubmission.validate_results_csv(df, dataset_name)
        
        if validation['valid']:
            return {
                'valid': True,
                'message': 'CSV format is valid',
                'essay_count': validation['essay_count'],
                'score_range': validation['score_range'],
                'columns_found': list(df.columns),
                'sample_data': df.head(3).to_dict('records')
            }
        else:
            return {
                'valid': False,
                'error': validation['error'],
                'required_columns': validation.get('required_columns', []),
                'columns_found': list(df.columns)
            }
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Validation error: {str(e)}")

@router.get("/submission-status/{submission_id}")
async def get_submission_status(submission_id: int):
    """Get status of a submission"""
    
    try:
        # Get submission details
        submission = DatabaseService.get_output_submission(submission_id)
        if not submission:
            raise HTTPException(status_code=404, detail="Submission not found")
        
        # Get evaluation results
        evaluation_results = DatabaseService.get_evaluation_results_by_submission(submission_id)
        
        return {
            'submission_id': submission_id,
            'dataset_name': submission.dataset_name,
            'submitter_name': submission.submitter_name,
            'status': submission.status,
            'submitted_at': submission.submission_time.isoformat(),
            'processing_time': submission.processing_time,
            'evaluation_result': submission.evaluation_result,
            'detailed_evaluations': evaluation_results,
            'error_message': submission.error_message
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving status: {str(e)}")

@router.get("/recent")
async def get_recent_submissions(limit: int = 10):
    """Get recent submissions for monitoring"""
    
    try:
        submissions = DatabaseService.get_recent_submissions(limit)
        return {
            'submissions': submissions,
            'total_count': len(submissions)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving submissions: {str(e)}")

# Health check endpoint for this router
@router.get("/health")
async def submissions_health():
    """Health check for submissions service"""
    return {
        'service': 'submissions',
        'status': 'healthy',
        'timestamp': datetime.now().isoformat()
    }


@router.post("/upload-benchmark-results")
async def upload_benchmark_results(
    model_name: str = Form(...),
    submitter_name: str = Form(...),
    submitter_email: str = Form(...),
    model_description: Optional[str] = Form(None),
    paper_url: Optional[str] = Form(None),
    results_zip: UploadFile = File(..., description="ZIP file containing CSV results for all 12 datasets")
):
    """Upload complete benchmark results as ZIP file containing CSV files for all 12 datasets"""
    
    try:
        print(f"📤 Processing complete benchmark submission from {submitter_name}")
        print(f"   Model: {model_name}")
        print(f"   ZIP file: {results_zip.filename}")
        start_time = time.time()
        
        # Validate ZIP file
        if not results_zip.filename.endswith('.zip'):
            raise HTTPException(status_code=400, detail="File must be a ZIP archive")
        
        # Get list of required datasets
        required_datasets = set(dataset_manager.datasets_config.keys())
        print(f"   📋 Required datasets: {len(required_datasets)} total")
        
        # Create temporary directory to extract ZIP
        with tempfile.TemporaryDirectory() as temp_dir:
            # Save uploaded ZIP file
            zip_path = os.path.join(temp_dir, "results.zip")  # Use os.path.join instead of Path
            with open(zip_path, "wb") as f:
                content = await results_zip.read()
                f.write(content)
            
            print(f"   💾 Saved ZIP to temporary location: {len(content)} bytes")
            
            # Extract and validate ZIP contents
            try:
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(temp_dir)
                    print(f"   📁 Extracted ZIP contents")
            except zipfile.BadZipFile:
                raise HTTPException(status_code=400, detail="Invalid ZIP file")
            
            # Find CSV files in extracted content
            csv_files = {}
            for root, dirs, files in os.walk(temp_dir):
                for file in files:
                    if file.endswith('.csv'):
                        file_path = os.path.join(root, file)  # Use os.path.join
                        # Extract dataset name from filename
                        dataset_name = file.replace('.csv', '').replace('_results', '')
                        if dataset_name in required_datasets:
                            csv_files[dataset_name] = file_path
            
            print(f"   📁 Found CSV files for datasets: {list(csv_files.keys())}")
            
            # Check if all required datasets are present
            missing_datasets = required_datasets - set(csv_files.keys())
            if missing_datasets:
                raise HTTPException(
                    status_code=400, 
                    detail=f"Missing CSV files for datasets: {list(missing_datasets)}. Required: {list(required_datasets)}"
                )
            
            print(f"   ✅ All required datasets found!")
            
            # Validate each CSV file
            validated_results = {}
            for dataset_name, csv_path in csv_files.items():
                try:
                    df = pd.read_csv(csv_path)
                    print(f"   📊 {dataset_name}: {df.shape[0]} rows, {df.shape[1]} columns")
                    
                    # Validate CSV format
                    validation = ResultsSubmission.validate_results_csv(df, dataset_name)
                    if not validation['valid']:
                        raise HTTPException(
                            status_code=400, 
                            detail=f"Invalid CSV format for {dataset_name}: {validation['error']}"
                        )
                    
                    validated_results[dataset_name] = {
                        'dataframe': df,
                        'validation': validation
                    }
                    print(f"   ✅ {dataset_name}: Validated {validation['essay_count']} predictions")
                    
                except Exception as e:
                    print(f"   ❌ Error validating {dataset_name}: {e}")
                    raise HTTPException(
                        status_code=400, 
                        detail=f"Error processing {dataset_name}.csv: {str(e)}"
                    )
            
            print(f"   🎯 All {len(validated_results)} datasets validated successfully!")
            
            # Create main submission record
            main_submission_data = {
                "dataset_name": "BENCHMARK_COMPLETE",  # Special dataset name for complete benchmark
                "submitter_name": submitter_name,
                "submitter_email": submitter_email,
                "file_path": f"benchmarks/{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip",
                "file_format": "zip",
                "status": "processing",
                "description": f"Complete benchmark: {model_name} - {model_description}" if model_description else f"Complete benchmark: {model_name}"
            }
            
            main_submission_dict = DatabaseService.save_output_submission(main_submission_data)
            main_submission_id = main_submission_dict["id"]
            print(f"   💾 Created main benchmark submission {main_submission_id}")
            
            # Process each dataset
            dataset_results = []
            total_metrics = []
            failed_datasets = []
            
            for dataset_name, results in validated_results.items():
                try:
                    print(f"   🧮 Processing {dataset_name}...")
                    
                    # Load original dataset
                    original_essays = dataset_manager.load_dataset_for_evaluation(dataset_name, sample_size=1000)
                    human_scores_lookup = {essay['essay_id']: essay['human_score'] for essay in original_essays}
                    
                    # Match predictions with human scores
                    results_df = results['dataframe']
                    matched_data = []
                    
                    for _, row in results_df.iterrows():
                        essay_id = row['essay_id']
                        predicted_score = row['predicted_score']
                        
                        if essay_id in human_scores_lookup:
                            human_score = human_scores_lookup[essay_id]
                            matched_data.append({
                                'essay_id': essay_id,
                                'human_score': human_score,
                                'predicted_score': predicted_score
                            })
                    
                    if not matched_data:
                        failed_datasets.append(dataset_name)
                        print(f"   ❌ {dataset_name}: No matching essays found")
                        continue
                    
                    # Calculate metrics
                    human_scores = [item['human_score'] for item in matched_data]
                    predicted_scores = [item['predicted_score'] for item in matched_data]
                    metrics = calculate_evaluation_metrics(human_scores, predicted_scores)
                    
                    # Create individual dataset submission
                    dataset_submission_data = {
                        "dataset_name": dataset_name,
                        "submitter_name": submitter_name,
                        "submitter_email": submitter_email,
                        "file_path": f"benchmarks/{model_name}_{dataset_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        "file_format": "csv",
                        "status": "completed",
                        "description": f"Benchmark result for {dataset_name} - {model_name}"
                    }
                    
                    dataset_submission_dict = DatabaseService.save_output_submission(dataset_submission_data)
                    dataset_submission_id = dataset_submission_dict["id"]
                    
                    # Create evaluation result
                    evaluation_result = {
                        'quadratic_weighted_kappa': metrics['qwk'],
                        'pearson_correlation': metrics['pearson'],
                        'spearman_correlation': metrics['spearman'],
                        'mean_absolute_error': metrics['mae'],
                        'root_mean_squared_error': metrics['rmse'],
                        'f1_score': metrics['f1'],
                        'accuracy': metrics['accuracy'],
                        'essays_evaluated': len(matched_data),
                        'detailed_metrics': {
                            'matched_essays': len(matched_data),
                            'submitted_essays': len(results_df),
                            'match_rate': len(matched_data) / len(results_df),
                            'model_name': model_name,
                            'submission_method': 'benchmark_zip'
                        }
                    }
                    
                    # Update dataset submission with results
                    DatabaseService.update_output_submission_status(
                        dataset_submission_id, 
                        "completed", 
                        evaluation_result=evaluation_result
                    )
                    
                    # Save detailed evaluation
                    DatabaseService.save_evaluation_result(dataset_submission_id, dataset_name, evaluation_result)
                    
                    dataset_results.append({
                        'dataset_name': dataset_name,
                        'submission_id': dataset_submission_id,
                        'metrics': metrics,
                        'essays_evaluated': len(matched_data)
                    })
                    
                    total_metrics.append(metrics)
                    print(f"   ✅ {dataset_name}: QWK={metrics['qwk']:.3f}, Essays={len(matched_data)}")
                    
                except Exception as e:
                    failed_datasets.append(dataset_name)
                    print(f"   ❌ Error processing {dataset_name}: {e}")
            
            # Calculate overall benchmark results
            processing_time = time.time() - start_time
            successful_datasets = len(total_metrics)
            
            if successful_datasets >= 10:  # Allow up to 2 failed datasets
                # Calculate aggregate metrics
                avg_qwk = sum(m['qwk'] for m in total_metrics) / len(total_metrics)
                avg_pearson = sum(m['pearson'] for m in total_metrics) / len(total_metrics)
                avg_mae = sum(m['mae'] for m in total_metrics) / len(total_metrics)
                avg_accuracy = sum(m['accuracy'] for m in total_metrics) / len(total_metrics)
                total_essays = sum(r['essays_evaluated'] for r in dataset_results)
                
                # Create benchmark summary
                benchmark_result = {
                    'avg_quadratic_weighted_kappa': avg_qwk,
                    'avg_pearson_correlation': avg_pearson,
                    'avg_mean_absolute_error': avg_mae,
                    'avg_accuracy': avg_accuracy,
                    'datasets_completed': successful_datasets,
                    'total_datasets': len(required_datasets),
                    'total_essays_evaluated': total_essays,
                    'failed_datasets': failed_datasets,
                    'detailed_results': {r['dataset_name']: r['metrics'] for r in dataset_results},
                    'submission_method': 'complete_benchmark_zip'
                }
                
                # Update main submission
                DatabaseService.update_output_submission_status(
                    main_submission_id, 
                    "completed", 
                    evaluation_result=benchmark_result,
                    processing_time=processing_time
                )
                
                print(f"🎉 Benchmark completed! Average QWK: {avg_qwk:.3f} across {successful_datasets} datasets")
                
                return {
                    'message': 'Benchmark evaluation completed successfully!',
                    'submission_id': main_submission_id,
                    'model_name': model_name,
                    'benchmark_results': {
                        'avg_quadratic_weighted_kappa': avg_qwk,
                        'avg_pearson_correlation': avg_pearson,
                        'avg_mean_absolute_error': avg_mae,
                        'avg_accuracy': avg_accuracy,
                        'datasets_completed': successful_datasets,
                        'total_datasets': len(required_datasets),
                        'completion_rate': (successful_datasets / len(required_datasets)) * 100,
                        'total_essays_evaluated': total_essays
                    },
                    'dataset_results': dataset_results,
                    'failed_datasets': failed_datasets,
                    'processing_time_seconds': processing_time,
                    'eligible_for_leaderboard': True
                }
            else:
                # Too many failed datasets
                DatabaseService.update_output_submission_status(
                    main_submission_id, 
                    "failed", 
                    error_message=f"Insufficient successful evaluations: {successful_datasets}/{len(required_datasets)}. Failed datasets: {failed_datasets}"
                )
                
                raise HTTPException(
                    status_code=400, 
                    detail=f"Benchmark requires at least 10 successful dataset evaluations. Only {successful_datasets} succeeded. Failed: {failed_datasets}"
                )
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error processing benchmark submission: {e}")
        raise HTTPException(status_code=500, detail=f"Benchmark processing error: {str(e)}")

@router.get("/benchmark-template")
async def get_benchmark_template():
    """Get information about required ZIP structure for benchmark submission"""
    
    required_datasets = list(dataset_manager.datasets_config.keys())
    
    return {
        'description': 'Complete benchmark submission requires a ZIP file with CSV results for all datasets',
        'required_datasets': required_datasets,
        'zip_structure': {
            'required_files': [f"{dataset}.csv" for dataset in required_datasets],
            'csv_format': {
                'columns': ['essay_id', 'predicted_score'],
                'example': 'essay_id,predicted_score\nASAP-AES_0,3.5\nASAP-AES_1,4.2'
            }
        },
        'instructions': [
            '1. Download all datasets using /api/datasets/download/all',
            '2. Run your model on each dataset',
            '3. Create CSV files: ASAP-AES.csv, rice_chem.csv, etc.',
            '4. Create ZIP file containing all 12 CSV files',
            '5. Upload ZIP file using /submissions/upload-benchmark-results',
            '6. Results will appear on leaderboard only after complete evaluation'
        ],
        'example_zip_contents': [
            'ASAP-AES.csv',
            'ASAP-SAS.csv', 
            'rice_chem.csv',
            'CSEE.csv',
            '... (all 12 datasets)'
        ]
    }