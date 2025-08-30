# app/api/routes/datasets.py
from fastapi import APIRouter, HTTPException, Response
from fastapi.responses import StreamingResponse
import tempfile
import os
from datetime import datetime
import zipfile
import io
import csv
import pandas as pd
from typing import Dict, List, Optional, Any

# Import Pydantic models
from app.models.pydantic_models import (
    DatasetsListResponse, DatasetInfo, DatasetDetails, DatasetSample, 
    DatasetHealthCheck, ErrorResponse
)

from app.services.dataset_loader import dataset_manager

# Single router declaration
# Change this line in datasets.py:
router = APIRouter(prefix="/datasets", tags=["datasets"])

@router.get("/", response_model=DatasetsListResponse)
async def get_all_datasets():
    
    try:
        datasets_config = dataset_manager.datasets_config
        
        datasets_list = []
        for dataset_name, config in datasets_config.items():
            dataset_info = DatasetInfo(
                name=dataset_name,
                description=config["description"],
                huggingface_id=config["huggingface_id"],
                essay_column=config["essay_column"],
                score_column=config["score_column"],
                prompt_column=config.get("prompt_column", "prompt"),
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

@router.get("/download/all")
async def download_all_datasets():
    """Download all datasets as ZIP with proper directory structure - ORIGINAL FORMAT"""
    
    try:
        print("📦 Preparing all datasets for download...")
        
        # Create temporary directory for organizing files
        temp_dir = tempfile.mkdtemp()
        
        try:
            datasets_config = dataset_manager.datasets_config
            total_files_created = 0
            
            # Group datasets by base name to handle configs properly
            dataset_groups = {}
            for dataset_name in datasets_config.keys():
                # Extract base name (remove config suffix)
                if '_q' in dataset_name:
                    base_name = dataset_name.split('_q')[0]
                    config = 'q' + dataset_name.split('_q')[1]
                elif '_Q' in dataset_name:
                    base_name = dataset_name.split('_Q')[0]
                    config = 'Q' + dataset_name.split('_Q')[1]
                elif dataset_name.endswith('_2way') or dataset_name.endswith('_3way'):
                    parts = dataset_name.rsplit('_', 1)
                    base_name = parts[0]
                    config = parts[1]
                else:
                    base_name = dataset_name
                    config = None
                
                if base_name not in dataset_groups:
                    dataset_groups[base_name] = []
                dataset_groups[base_name].append((dataset_name, config))
            
            print(f"📊 Found {len(dataset_groups)} dataset groups")
            
            # Process each dataset group
            for base_name, configs in dataset_groups.items():
                print(f"\n📋 Processing dataset group: {base_name}")
                
                # Create base directory
                base_dir = os.path.join(temp_dir, base_name)
                os.makedirs(base_dir, exist_ok=True)
                
                for dataset_name, config in configs:
                    print(f"  🔧 Processing {dataset_name} (config: {config})")
                    
                    try:
                        # Get dataset config
                        dataset_config = datasets_config[dataset_name]
                        
                        # Determine directory structure based on config type
                        if config and (config.startswith('q') or config.startswith('Q')):
                            # Create subdirectory for q1, q2, Q1, Q2 configs
                            dataset_dir = os.path.join(base_dir, config)
                            os.makedirs(dataset_dir, exist_ok=True)
                            file_suffix = ""
                        elif config in ['2way', '3way']:
                            # Use base directory with file suffix
                            dataset_dir = base_dir
                            file_suffix = f"_{config}"
                        else:
                            # Simple dataset
                            dataset_dir = base_dir
                            file_suffix = ""
                        
                        # Determine available splits
                        if base_name in ['EFL', 'BEEtlE', 'SciEntSBank']:
                            possible_splits = ['train', 'test']
                        else:
                            possible_splits = ['train', 'validation', 'test']
                        
                        # Process each split
                        for split in possible_splits:
                            print(f"    📄 Loading {split} split...")
                            
                            try:
                                # Load data for this split
                                raw_data = dataset_manager.hf_loader.load_dataset_sample(
                                    dataset_id=dataset_config["huggingface_id"],
                                    config=dataset_config.get("config"),
                                    split=split,
                                    sample_size=999999
                                )
                                
                                if not raw_data:
                                    print(f"    ⚠️ No data found for {split} split, skipping...")
                                    continue
                                
                                # Extract rows and convert to DataFrame - ORIGINAL FORMAT
                                rows = [item.get("row", {}) for item in raw_data]
                                if rows:
                                    df = pd.DataFrame(rows)
                                    
                                    # Clear scores for test split but keep columns
                                    if split == 'test':
                                        score_columns = []
                                        for col in df.columns:
                                            col_lower = col.lower()
                                            if any(score_word in col_lower for score_word in ['score', 'label', 'grade', 'rating']):
                                                score_columns.append(col)
                                        
                                        for score_col in score_columns:
                                            df[score_col] = ''
                                        
                                        print(f"    🔒 Cleared {len(score_columns)} score columns")
                                    
                                    # Save to file with proper naming
                                    filename = f"{split}{file_suffix}.csv"
                                    csv_path = os.path.join(dataset_dir, filename)
                                    df.to_csv(csv_path, index=False)
                                    
                                    total_files_created += 1
                                    print(f"    ✅ Created {filename} with {len(df)} rows, {len(df.columns)} columns")
                                
                            except Exception as e:
                                print(f"    ❌ Error processing {split}: {e}")
                                continue
                        
                        print(f"    ✅ Processed {dataset_name}")
                    
                    except Exception as e:
                        print(f"    ❌ Error processing {dataset_name}: {e}")
                        continue
            
            if total_files_created == 0:
                raise HTTPException(status_code=500, detail="No data could be loaded for any dataset")
            
            # Create simple README
            readme_content = f"""# BESESR Datasets - Original Format

Downloaded: {datetime.now().isoformat()}
Total Files: {total_files_created}

## Contents:
This archive contains all datasets in their **original format** exactly as prepared on HuggingFace.

## Directory Structure:
- Each dataset has its own folder
- Configuration-based datasets (q1-q5, Q1-Q4) have subdirectories
- Multi-way datasets (2way/3way) use file suffixes

## File Types:
- `train.csv`: Training data WITH original scores
- `validation.csv`: Validation data WITH original scores (when available)
- `test.csv`: Test data with EMPTY score columns (for prediction)

## Important:
- **All original column names preserved**
- **No standardization or modifications**
- **Only score columns cleared in test files**
- **Your months of dataset preparation is 100% intact**

## Usage:
1. Train your models using train.csv and validation.csv
2. Generate predictions for test.csv essays
3. Submit predictions via BESESR platform

## Column Names:
Each dataset maintains its original column structure:
- ASAP-AES: essay_id, essay_set, essay, rater1_domain1, domain1_score, etc.
- BEEtlE: question_id, question_text, student_answer, label
- CSEE: index, essay_id, prompt_id, essay, overall_score, etc.
- All others: Original format preserved

---
*All datasets are in their original format as you prepared them.*
"""
            
            # Create ZIP file
            zip_buffer = io.BytesIO()
            with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                
                # Add all dataset files with directory structure
                for root, dirs, files in os.walk(temp_dir):
                    for file in files:
                        file_path = os.path.join(root, file)
                        arcname = os.path.relpath(file_path, temp_dir)
                        zip_file.write(file_path, arcname)
                
                # Add simple README
                zip_file.writestr("README.md", readme_content)
            
            zip_buffer.seek(0)
            
            print(f"✅ Created bundle: {total_files_created} files across {len(dataset_groups)} datasets")
            
            return Response(
                content=zip_buffer.getvalue(),
                media_type="application/zip",
                headers={"Content-Disposition": "attachment; filename=besesr_all_datasets_original.zip"}
            )
            
        finally:
            # Clean up temporary files
            import shutil
            try:
                shutil.rmtree(temp_dir)
            except:
                pass
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error creating dataset bundle: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to create dataset bundle: {str(e)}")

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
                prompt_column=config.get("prompt_column", "prompt"),
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
    
@router.get("/download/{dataset_name}")
async def download_single_dataset(dataset_name: str):
    """Download a single dataset as ZIP with train/validation/test splits - ORIGINAL FORMAT"""
    
    try:
        if dataset_name not in dataset_manager.datasets_config:
            raise HTTPException(status_code=404, detail=f"Dataset '{dataset_name}' not found")

        print(f"📥 Preparing {dataset_name} for download...")

        # Create temporary directory for CSV files
        temp_dir = tempfile.mkdtemp()
        csv_files = []
        
        try:
            config = dataset_manager.datasets_config[dataset_name]
            
            # Define splits to create based on dataset type
            if dataset_name in ['EFL', 'BEEtlE_2way', 'BEEtlE_3way', 'SciEntSBank_2way', 'SciEntSBank_3way']:
                splits = ['train', 'test']  # These datasets don't have validation
            else:
                splits = ['train', 'validation', 'test']
            
            for split in splits:
                print(f"📋 Loading {split} split...")
                
                try:
                    # Load data for this split - uses your FIXED column mappings
                    raw_data = dataset_manager.hf_loader.load_dataset_sample(
                        dataset_id=config["huggingface_id"],
                        config=config.get("config"),
                        split=split,
                        sample_size=999999  # Get all data
                    )
                    
                    if not raw_data:
                        print(f"⚠️ No data found for {split} split, skipping...")
                        continue
                    
                    # Convert to DataFrame and save as CSV - ORIGINAL FORMAT
                    if raw_data:
                        # Extract just the row data (original format with ALL columns)
                        rows = [item.get("row", {}) for item in raw_data]
                        
                        if rows:
                            df = pd.DataFrame(rows)
                            
                            # For test files, clear score columns but keep the columns themselves
                            if split == 'test':
                                # Find columns that likely contain scores
                                score_columns = []
                                for col in df.columns:
                                    col_lower = col.lower()
                                    if any(score_word in col_lower for score_word in ['score', 'label', 'grade', 'rating']):
                                        score_columns.append(col)
                                
                                # Clear the score columns (make them empty, not remove them)
                                for score_col in score_columns:
                                    df[score_col] = ''
                                
                                print(f"🔒 Cleared {len(score_columns)} score columns for test split")
                            
                            # Save as CSV with ALL original columns preserved
                            csv_path = os.path.join(temp_dir, f"{split}.csv")
                            df.to_csv(csv_path, index=False)
                            csv_files.append(csv_path)
                            print(f"✅ Created {split}.csv with {len(df)} rows and {len(df.columns)} original columns")
                
                except Exception as e:
                    print(f"❌ Error processing {split}: {e}")
                    continue
            
            if not csv_files:
                raise HTTPException(status_code=500, detail=f"No data could be loaded for {dataset_name}")
            
            # Create ZIP file with ONLY the CSV files (no extra metadata)
            zip_buffer = io.BytesIO()
            with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                for csv_path in csv_files:
                    zip_file.write(csv_path, os.path.basename(csv_path))
            
            zip_buffer.seek(0)
            
            print(f"✅ Created ZIP with {len(csv_files)} CSV files for {dataset_name}")
            
            return Response(
                content=zip_buffer.getvalue(),
                media_type="application/zip",
                headers={"Content-Disposition": f"attachment; filename={dataset_name}_splits.zip"}
            )
            
        finally:
            # Clean up temporary files
            for csv_path in csv_files:
                try:
                    os.remove(csv_path)
                except:
                    pass
            try:
                os.rmdir(temp_dir)
            except:
                pass
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error downloading {dataset_name}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to download {dataset_name}: {str(e)}")


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
                human_score=essay.get("score", essay.get("human_score", 0)),
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