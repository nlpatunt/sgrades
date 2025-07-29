from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from typing import List, Dict, Any, Optional
import pandas as pd
import io
import uuid
from datetime import datetime
import zipfile

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
        print(f"📤 Processing results submission from {submitter_name}")
        
        # Generate submission ID
        submission_id = f"submission_{uuid.uuid4().hex[:8]}"
        
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
                'validation': validation
            }
            
            print(f"   ✅ Validated {dataset_name}: {validation['essay_count']} predictions")
        
        if not dataset_results:
            raise HTTPException(status_code=400, detail="No valid dataset results found")
        
        # Save model submission
        model_data = {
            'id': submission_id,
            'model_name': model_name,
            'model_type': 'csv_submission',
            'api_endpoint': None,
            'huggingface_model_id': None,
            'submitter_name': submitter_name,
            'submitter_email': submitter_email,
            'model_description': model_description,
            'paper_url': paper_url,
            'status': 'evaluating'
        }
        
        model = DatabaseService.save_model_submission(model_data)
        print(f"💾 Saved model submission: {submission_id}")
        
        # Process each dataset
        total_metrics = []
        
        for dataset_name, results in dataset_results.items():
            try:
                print(f"🧮 Calculating metrics for {dataset_name}...")
                
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
                    raise Exception(f"No matching essays found between submission and original dataset")
                
                # Extract scores for metrics calculation
                human_scores = [item['human_score'] for item in matched_data]
                predicted_scores = [item['predicted_score'] for item in matched_data]
                
                # Calculate metrics
                metrics = calculate_evaluation_metrics(human_scores, predicted_scores)
                
                # Save evaluation result
                evaluation_result = {
                    'quadratic_weighted_kappa': metrics['qwk'],
                    'pearson_correlation': metrics['pearson'],
                    'spearman_correlation': metrics['spearman'],
                    'mean_absolute_error': metrics['mae'],
                    'root_mean_squared_error': metrics['rmse'],
                    'f1_score': metrics['f1'],
                    'accuracy': metrics['accuracy'],
                    'essays_evaluated': len(matched_data),
                    'evaluation_duration': 0.0,  # Instant for CSV submission
                    'status': 'completed',
                    'detailed_metrics': {
                        'matched_essays': len(matched_data),
                        'submitted_essays': len(results_df),
                        'match_rate': len(matched_data) / len(results_df),
                        'score_distribution': metrics.get('distribution', {}),
                        'submission_method': 'csv_upload'
                    }
                }
                
                DatabaseService.save_evaluation_result(submission_id, dataset_name, evaluation_result)
                total_metrics.append(metrics)
                
                print(f"   ✅ {dataset_name}: QWK={metrics['qwk']:.3f}, Pearson={metrics['pearson']:.3f}, Essays={len(matched_data)}")
                
            except Exception as e:
                print(f"   ❌ Error processing {dataset_name}: {e}")
                # Save failed result
                DatabaseService.save_evaluation_result(submission_id, dataset_name, {
                    'status': 'failed',
                    'error_message': str(e),
                    'essays_evaluated': 0
                })
        
        # Update model status
        if total_metrics:
            DatabaseService.update_model_status(submission_id, "completed")
            avg_qwk = sum(m['qwk'] for m in total_metrics) / len(total_metrics)
            print(f"🎉 Submission completed! Average QWK: {avg_qwk:.3f}")
        else:
            DatabaseService.update_model_status(submission_id, "failed")
        
        return {
            'submission_id': submission_id,
            'model_name': model_name,
            'datasets_processed': len(dataset_results),
            'successful_evaluations': len(total_metrics),
            'average_qwk': sum(m['qwk'] for m in total_metrics) / len(total_metrics) if total_metrics else 0,
            'message': 'Results submitted and evaluated successfully!',
            'leaderboard_url': '/leaderboard/'
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
            '4. Upload all CSV files using the submission form',
            '5. Platform will calculate metrics and update leaderboard'
        ]
    }