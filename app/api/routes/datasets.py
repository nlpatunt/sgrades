# app/api/routes/datasets.py
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from typing import List, Dict, Any
from datetime import datetime
import zipfile
import io
import csv
import pandas as pd

# Import Pydantic models
from app.models.pydantic_models import (
    DatasetsListResponse, DatasetInfo, DatasetDetails, DatasetSample, 
    DatasetHealthCheck, ErrorResponse
)

# Import the dynamic dataset manager instead of static config
from app.services.dataset_loader import dataset_manager

router = APIRouter(prefix="/datasets", tags=["datasets"])

@router.get("/", response_model=DatasetsListResponse)
async def get_all_datasets():
    """Get all available datasets from dynamic HuggingFace configuration"""
    
    try:
        # Get dynamic dataset configuration
        datasets_config = dataset_manager.datasets_config
        
        # Format for API response
        datasets_list = []
        for dataset_name, config in datasets_config.items():
            dataset_info = DatasetInfo(
                name=dataset_name,
                description=config["description"],
                huggingface_id=config["huggingface_id"],
                essay_column=config["essay_column"],
                score_column=config["score_column"],
                prompt_column=config["prompt_column"],
                score_range=config["score_range"],
                split=config["split"],
                status="active"
            )
            datasets_list.append(dataset_info)
        
        return DatasetsListResponse(
            datasets=datasets_list,
            total_count=len(datasets_list),
            data_source="dynamic_huggingface",
            last_updated=datetime.now().isoformat()
        )
        
    except Exception as e:
        print(f"❌ Error getting datasets: {e}")
        return DatasetsListResponse(
            datasets=[],
            total_count=0,
            data_source="error",
            last_updated=datetime.now().isoformat()
        )

@router.get("/{dataset_name}", response_model=DatasetDetails)
async def get_dataset_details(dataset_name: str):
    """Get detailed information about a specific dataset"""
    
    try:
        datasets_config = dataset_manager.datasets_config
        
        if dataset_name not in datasets_config:
            raise HTTPException(
                status_code=404,
                detail=f"Dataset '{dataset_name}' not found. Available datasets: {list(datasets_config.keys())}"
            )
        
        config = datasets_config[dataset_name]
        
        # Try to get real dataset info from HuggingFace
        hf_info = dataset_manager.hf_loader.get_dataset_info(config["huggingface_id"]) if hasattr(dataset_manager.hf_loader, 'get_dataset_info') else None
        
        from app.models.pydantic_models import DatasetConfiguration
        
        return DatasetDetails(
            name=dataset_name,
            description=config["description"],
            huggingface_id=config["huggingface_id"],
            configuration=DatasetConfiguration(
                essay_column=config["essay_column"],
                score_column=config["score_column"],
                prompt_column=config["prompt_column"],
                score_range=config["score_range"],
                split=config["split"]
            ),
            huggingface_info=hf_info,
            sample_available=True,
            status="active"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error getting dataset details for {dataset_name}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{dataset_name}/sample", response_model=DatasetSample)
async def get_dataset_sample(dataset_name: str, size: int = 5):
    """Get a sample of essays from the dataset"""
    
    try:
        print(f"📊 Getting sample from {dataset_name}")
        
        # Load sample essays
        sample_essays = dataset_manager.load_dataset_for_evaluation(
            dataset_name, 
            sample_size=min(size, 10)  # Limit sample size
        )
        
        if not sample_essays:
            raise HTTPException(
                status_code=404,
                detail=f"Could not load sample from {dataset_name}"
            )
        
        # Format sample for API response (remove full essay text for privacy)
        from app.models.pydantic_models import EssayPreview
        
        formatted_sample = []
        for essay in sample_essays:
            essay_preview = EssayPreview(
                essay_id=essay["essay_id"],
                essay_preview=essay["essay_text"][:200] + "..." if len(essay["essay_text"]) > 200 else essay["essay_text"],
                prompt=essay["prompt"],
                human_score=essay["human_score"],
                score_range=essay["score_range"],
                word_count=len(essay["essay_text"].split()),
                metadata=essay.get("metadata", {})
            )
            formatted_sample.append(essay_preview)
        
        return DatasetSample(
            dataset_name=dataset_name,
            sample_size=len(formatted_sample),
            requested_size=size,
            essays=formatted_sample,
            loaded_at=datetime.now().isoformat()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error getting sample from {dataset_name}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/download/all")
async def download_all_datasets():
    """Download all datasets as a ZIP file for local evaluation"""
    
    try:
        print("📦 Preparing dataset bundle for download...")
        
        # Create ZIP file in memory
        zip_buffer = io.BytesIO()
        
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            
            datasets_config = dataset_manager.datasets_config
            dataset_info = []
            
            for dataset_name, config in datasets_config.items():
                try:
                    print(f"📥 Adding {dataset_name} to bundle...")
                    
                    # Load dataset essays
                    essays = dataset_manager.load_dataset_for_evaluation(
                        dataset_name, 
                        sample_size=100  # Download more essays for evaluation
                    )
                    
                    if essays:
                        # Create CSV content
                        csv_content = []
                        for essay in essays:
                            csv_content.append({
                                'essay_id': essay['essay_id'],
                                'dataset_name': essay['dataset_name'],
                                'essay_text': essay['essay_text'],
                                'prompt': essay['prompt'],
                                'human_score': essay['human_score'],
                                'score_range_min': essay['score_range'][0],
                                'score_range_max': essay['score_range'][1]
                            })
                        
                        # Convert to CSV string
                        csv_buffer = io.StringIO()
                        if csv_content:
                            fieldnames = csv_content[0].keys()
                            writer = csv.DictWriter(csv_buffer, fieldnames=fieldnames)
                            writer.writeheader()
                            writer.writerows(csv_content)
                        
                        # Add to ZIP
                        csv_string = csv_buffer.getvalue()
                        zip_file.writestr(f"{dataset_name}.csv", csv_string)
                        
                        # Track dataset info
                        dataset_info.append({
                            'dataset_name': dataset_name,
                            'essay_count': len(essays),
                            'description': config['description'],
                            'score_range': essay['score_range'],
                            'huggingface_id': config['huggingface_id']
                        })
                        
                        print(f"   ✅ Added {len(essays)} essays from {dataset_name}")
                    
                except Exception as e:
                    print(f"   ❌ Error processing {dataset_name}: {e}")
                    continue
            
            # Add README file
            readme_content = f"""# BESESR Datasets Bundle

Downloaded: {datetime.now().isoformat()}
Total Datasets: {len(dataset_info)}

## Dataset Information:
"""
            
            for info in dataset_info:
                readme_content += f"""
### {info['dataset_name']}
- Description: {info['description']}
- Essays: {info['essay_count']}
- Score Range: {info['score_range'][0]} - {info['score_range'][1]}
- HuggingFace ID: {info['huggingface_id']}
"""
            
            readme_content += f"""

## Instructions:

1. Each CSV file contains essays from one dataset
2. Run your model on each dataset
3. Create results CSV with columns: essay_id, predicted_score
4. Submit results via the platform upload interface

## CSV Format for Each Dataset:
- essay_id: Unique identifier for each essay
- dataset_name: Name of the dataset
- essay_text: The essay content to evaluate
- prompt: The writing prompt/question
- human_score: Gold standard human score
- score_range_min/max: Valid score range for this dataset

## Expected Output Format:
Create one CSV file per dataset with columns:
- essay_id: Same as input
- predicted_score: Your model's predicted score

Example: ASAP-AES_results.csv
essay_id,predicted_score
ASAP-AES_0,3.5
ASAP-AES_1,4.2
...

## Submission:
1. Visit the BESESR platform
2. Go to "Submit Results" section
3. Upload all your results CSV files
4. Fill in model information
5. Submit for evaluation and leaderboard inclusion
"""
            
            zip_file.writestr("README.md", readme_content)
            
            # Add dataset summary
            summary_csv = io.StringIO()
            summary_writer = csv.DictWriter(summary_csv, fieldnames=['dataset_name', 'essay_count', 'score_range_min', 'score_range_max', 'description'])
            summary_writer.writeheader()
            for info in dataset_info:
                summary_writer.writerow({
                    'dataset_name': info['dataset_name'],
                    'essay_count': info['essay_count'],
                    'score_range_min': info['score_range'][0],
                    'score_range_max': info['score_range'][1],
                    'description': info['description']
                })
            
            zip_file.writestr("dataset_summary.csv", summary_csv.getvalue())
        
        zip_buffer.seek(0)
        
        print(f"✅ Dataset bundle created with {len(dataset_info)} datasets")
        
        # Return ZIP file
        return StreamingResponse(
            io.BytesIO(zip_buffer.read()),
            media_type="application/zip",
            headers={"Content-Disposition": f"attachment; filename=besesr_datasets_{datetime.now().strftime('%Y%m%d')}.zip"}
        )
        
    except Exception as e:
        print(f"❌ Error creating dataset bundle: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to create dataset bundle: {str(e)}")

@router.get("/download/{dataset_name}")
async def download_single_dataset(dataset_name: str):
    """Download a single dataset as CSV"""
    
    try:
        if dataset_name not in dataset_manager.datasets_config:
            raise HTTPException(status_code=404, detail=f"Dataset '{dataset_name}' not found")
        
        print(f"📥 Preparing {dataset_name} for download...")
        
        # Load dataset essays
        essays = dataset_manager.load_dataset_for_evaluation(dataset_name, sample_size=100)
        
        if not essays:
            raise HTTPException(status_code=500, detail=f"Could not load essays from {dataset_name}")
        
        # Create CSV content
        csv_content = []
        for essay in essays:
            csv_content.append({
                'essay_id': essay['essay_id'],
                'dataset_name': essay['dataset_name'],
                'essay_text': essay['essay_text'],
                'prompt': essay['prompt'],
                'human_score': essay['human_score'],
                'score_range_min': essay['score_range'][0],
                'score_range_max': essay['score_range'][1]
            })
        
        # Convert to CSV
        csv_buffer = io.StringIO()
        if csv_content:
            fieldnames = csv_content[0].keys()
            writer = csv.DictWriter(csv_buffer, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(csv_content)
        
        csv_string = csv_buffer.getvalue()
        
        print(f"✅ Prepared {len(essays)} essays from {dataset_name}")
        
        # Return CSV file
        return StreamingResponse(
            io.StringIO(csv_string),
            media_type="text/csv",
            headers={"Content-Disposition": f"attachment; filename={dataset_name}.csv"}
        )
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error downloading {dataset_name}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to download {dataset_name}: {str(e)}")

@router.get("/health/check", response_model=DatasetHealthCheck)
async def dataset_health_check():
    """Check if dataset loading system is working"""
    
    try:
        datasets_config = dataset_manager.datasets_config
        total_datasets = len(datasets_config)
        
        # Test loading one sample from first dataset
        first_dataset = list(datasets_config.keys())[0] if datasets_config else None
        test_sample = None
        
        if first_dataset:
            test_sample = dataset_manager.load_dataset_for_evaluation(first_dataset, sample_size=1)
        
        return DatasetHealthCheck(
            status="healthy",
            total_datasets_configured=total_datasets,
            authentication="authenticated" if dataset_manager.hf_loader.authenticated else "not_authenticated",
            test_dataset=first_dataset,
            test_sample_loaded=len(test_sample) > 0 if test_sample else False,
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        return DatasetHealthCheck(
            status="unhealthy",
            total_datasets_configured=0,
            authentication="error",
            timestamp=datetime.now().isoformat()
        )