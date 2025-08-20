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
router = APIRouter(prefix="/datasets", tags=["datasets"])

DATASET_COLUMN_CONFIGS = {
    'ASAP-AES': {
        'standardize': {
            'essay': 'essay_text',           # essay -> essay_text
            'essay_set': 'prompt'            # essay_set -> prompt
        },
        'preserve': [
            'essay_id', 'rater1_domain1', 'rater2_domain1', 'rater3_domain1', 
            'domain1_score', 'rater1_domain2', 'rater2_domain2', 'domain2_score',
            'rater1_trait1', 'rater1_trait2', 'rater1_trait3', 'rater1_trait4',
            'rater1_trait5', 'rater1_trait6', 'rater2_trait1', 'rater2_trait2',
            'rater2_trait3', 'rater2_trait4', 'rater2_trait5', 'rater2_trait6',
            'rater3_trait1', 'rater3_trait2', 'rater3_trait3', 'rater3_trait4',
            'rater3_trait5', 'rater3_trait6', 'prompt_rubric_source_text_source'
        ],
        'primary_score': 'domain1_score'
    },
    'ASAP2': {
        'standardize': {
            'full_text': 'essay_text',
            'assignment': 'prompt'
        },
        'preserve': [
            'essay_id', 'score', 'prompt_name', 'economically_disadvantaged',
            'student_disability_status', 'ell_status', 'race_ethnicity', 'gender',
            'source_text_1', 'source_text_2', 'source_text_3', 'source_text_4'
        ],
        'primary_score': 'score'
    },
    'ASAP-SAS': {
        'standardize': {
            'essay_text': 'essay_text',  # Already correct
            'essay_set': 'prompt'
        },
        'preserve': [
            'Id', 'Score1', 'Score2', 'prompt_&_source_essay_urls',
            'rubric_urls', 'image_urls'
        ],
        'primary_score': 'Score1'
    },
    'CSEE': {
        'standardize': {
            'essay': 'essay_text',
            'prompt': 'prompt'  # Already correct
        },
        'preserve': [
            'index', 'essay_id', 'prompt_id', 'overall_score', 'content_score',
            'language_score', 'structure_score', 'rubric_source'
        ],
        'primary_score': 'overall_score'
    },
    'BEEtlE': {
        'standardize': {
            'student_answer': 'essay_text',
            'question_text': 'prompt'
        },
        'preserve': [
            'question_id', 'label'
        ],
        'primary_score': 'label'
    },
    'SciEntSBank': {
        'standardize': {
            'student_answer': 'essay_text',
            'question_text': 'prompt'
        },
        'preserve': [
            'question_id', 'label'
        ],
        'primary_score': 'label'
    },
    'EFL': {
        'standardize': {
            'essay_link': 'essay_text',  # Note: might need special handling if it's actually a link
            'default_prompt': 'prompt'
        },
        'preserve': [
            'Essay Id', 'Domain', 'finetuned_prompt', 'R1', 'R2', 'R3', 'R4', 'R5',
            'R6', 'R7', 'R8', 'R9', 'R10', 'R11', 'R12', 'R13', 'R14', 'R15',
            '_Human_Mean', 'rubric_source'
        ],
        'primary_score': '_Human_Mean'
    },
    'grade_like_a_human_dataset_os': {
        'standardize': {
            'answer': 'essay_text',
            'question': 'prompt'
        },
        'preserve': [
            'question_id', 'sample_answer', 'criteria', 'sample_criteria',
            'full_points', 'id', 'score_1', 'score_2', 'score_3', 'score_outlier'
        ],
        'primary_score': 'score_1'
    },
    'Rice_Chem': {
        'standardize': {
            'student_response': 'essay_text',
            'Prompt': 'prompt'
        },
        'preserve': [
            'sis_id', 'Score', 'incorrect', 'correctly cites decreased electron electron repulsion',
            'relates decreased electron electron repulsion to decreased potential energy',
            # ... all the rubric columns would be preserved
            'question_id'
        ],
        'primary_score': 'Score'
    }
}

def basic_column_processing(raw_data: List[Dict], split: str) -> List[Dict]:
    """Fallback processing for datasets not in DATASET_COLUMN_CONFIGS"""
    processed_data = []
    
    for i, item in enumerate(raw_data):
        row = item.get("row", {})
        
        # Basic standardization - try to find essay and score columns
        essay_text = ''
        score_value = ''
        prompt_text = 'Default prompt'
        essay_id = f"unknown_{split}_{i}"
        
        # Try to find essay text
        essay_candidates = ['essay', 'full_text', 'essay_text', 'student_answer', 'answer', 'text']
        for candidate in essay_candidates:
            if candidate in row and row[candidate]:
                essay_text = str(row[candidate])
                break
        
        # Try to find ID
        id_candidates = ['essay_id', 'id', 'sis_id']
        for candidate in id_candidates:
            if candidate in row and row[candidate]:
                essay_id = str(row[candidate])
                break
        
        # Try to find prompt
        prompt_candidates = ['prompt', 'question', 'question_text', 'assignment', 'task']
        for candidate in prompt_candidates:
            if candidate in row and row[candidate]:
                prompt_text = str(row[candidate])
                break
        
        # Try to find score (only for train/validation)
        if split != 'test':
            score_candidates = ['score', 'grade', 'label', 'rating']
            for candidate in score_candidates:
                if candidate in row and row[candidate] is not None:
                    score_value = row[candidate]
                    break
        
        processed_row = {
            'essay_id': essay_id,
            'essay_text': essay_text,
            'prompt': prompt_text,
            'score': score_value if split != 'test' else '',
            'score_note': 'Basic processing - primary score detected' if split != 'test' else 'Empty for prediction task'
        }
        
        # Add all original columns as preserved
        for col, value in row.items():
            if col not in ['essay_id']:  # Avoid duplicating essay_id
                processed_row[f'original_{col}'] = value
        
        processed_data.append(processed_row)
    
    return processed_data

def process_dataset_columns(dataset_name: str, raw_data: List[Dict], split: str) -> List[Dict]:
    """
    Process dataset columns with smart mapping - standardize some, preserve others
    """
    base_dataset = dataset_name.split('_')[0] if '_' in dataset_name else dataset_name
    
    # Get column configuration
    if base_dataset not in DATASET_COLUMN_CONFIGS:
        # Fallback to basic standardization
        return basic_column_processing(raw_data, split)
    
    config = DATASET_COLUMN_CONFIGS[base_dataset]
    standardize_map = config.get('standardize', {})
    preserve_cols = config.get('preserve', [])
    primary_score = config.get('primary_score')
    
    processed_data = []
    
    for i, item in enumerate(raw_data):
        row = item.get("row", {})
        processed_row = {}
        
        # 1. Add standardized columns
        processed_row['essay_id'] = row.get('essay_id') or row.get('id') or row.get('sis_id') or f"{dataset_name}_{split}_{i}"
        
        # Map essay column
        essay_text = None
        for original_col, standard_col in standardize_map.items():
            if standard_col == 'essay_text' and original_col in row:
                essay_text = row[original_col]
                break
        processed_row['essay_text'] = str(essay_text) if essay_text else ''
        
        # Map prompt column  
        prompt = None
        for original_col, standard_col in standardize_map.items():
            if standard_col == 'prompt' and original_col in row:
                prompt = row[original_col]
                break
        processed_row['prompt'] = str(prompt) if prompt else f"Prompt for {dataset_name}"
        
        # Handle score column based on split
        if split == 'test':
            processed_row['score'] = ''
            processed_row['score_note'] = 'Empty for prediction task'
        else:
            score_value = row.get(primary_score) if primary_score and primary_score in row else ''
            processed_row['score'] = score_value
            processed_row['score_note'] = f'Primary score from {primary_score}' if primary_score else 'Training/validation score'
        
        # 2. Preserve all specified columns with original names
        for col in preserve_cols:
            if col in row:
                processed_row[col] = row[col]
        
        # 3. Add metadata
        processed_row['dataset_name'] = dataset_name
        processed_row['config'] = dataset_name.split('_')[-1] if '_' in dataset_name else 'default'
        
        # 4. Add mapping information
        processed_row['_mapping_info'] = {
            'essay_mapped_from': next((k for k, v in standardize_map.items() if v == 'essay_text'), 'N/A'),
            'prompt_mapped_from': next((k for k, v in standardize_map.items() if v == 'prompt'), 'N/A'),
            'primary_score_column': primary_score or 'N/A'
        }
        
        processed_data.append(processed_row)
    
    return processed_data

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
            
            # Define splits to create
            if dataset_name in ['EFL', 'BEEtlE_2way', 'BEEtlE_3way', 'SciEntSBank_2way', 'SciEntSBank_3way']:
                splits = ['train', 'test']  # These datasets don't have validation
            else:
                splits = ['train', 'validation', 'test']
            
            for split in splits:
                print(f"📋 Loading {split} split...")
                
                try:
                    # Load data for this split - this will use your FIXED column mappings
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
                        # Extract just the row data (original format)
                        rows = [item.get("row", {}) for item in raw_data]
                        
                        if rows:
                            df = pd.DataFrame(rows)
                            
                            # For test files, clear score columns but keep them
                            if split == 'test':
                                score_columns = [col for col in df.columns if 'score' in col.lower()]
                                for score_col in score_columns:
                                    df[score_col] = ''
                            
                            # Save as CSV
                            csv_path = os.path.join(temp_dir, f"{split}.csv")
                            df.to_csv(csv_path, index=False)
                            csv_files.append(csv_path)
                            print(f"✅ Created {split}.csv with {len(df)} rows - ORIGINAL COLUMNS")
                
                except Exception as e:
                    print(f"❌ Error processing {split}: {e}")
                    continue
            
            if not csv_files:
                raise HTTPException(status_code=500, detail=f"No data could be loaded for {dataset_name}")
            
            # Create ZIP file with ONLY the CSV files
            zip_buffer = io.BytesIO()
            with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                for csv_path in csv_files:
                    zip_file.write(csv_path, os.path.basename(csv_path))  # Just the CSV files!
            
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

# Add this import at the top of your datasets.py file
import json

@router.get("/download/all")
async def download_all_datasets():
    """Download all datasets as ZIP with proper directory structure and comprehensive metadata"""
    
    try:
        print("📦 Preparing structured dataset bundle with comprehensive metadata...")
        
        # Create temporary directory for organizing files
        temp_dir = tempfile.mkdtemp()
        
        try:
            datasets_config = dataset_manager.datasets_config
            dataset_info = []
            column_mappings = []
            
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
            total_files_created = 0
            
            # Process each dataset group
            for base_name, configs in dataset_groups.items():
                print(f"\n📋 Processing dataset group: {base_name}")
                
                # Create base directory
                base_dir = os.path.join(temp_dir, base_name)
                os.makedirs(base_dir, exist_ok=True)
                
                for dataset_name, config in configs:
                    print(f"  🔧 Processing {dataset_name} (config: {config})")
                    
                    try:
                        # Get dataset config for metadata
                        dataset_config = datasets_config[dataset_name]
                        
                        # Determine directory structure based on config type
                        if config and (config.startswith('q') or config.startswith('Q')):
                            dataset_dir = os.path.join(base_dir, config)
                            os.makedirs(dataset_dir, exist_ok=True)
                            file_suffix = ""
                        elif config in ['2way', '3way']:
                            dataset_dir = base_dir
                            file_suffix = f"_{config}"
                        else:
                            dataset_dir = base_dir
                            file_suffix = ""
                        
                        # Determine available splits
                        if base_name in ['EFL', 'BEEtlE', 'SciEntSBank']:
                            possible_splits = ['train', 'test']
                        else:
                            possible_splits = ['train', 'validation', 'test']
                        
                        # Collect column mapping info
                        column_mappings.append({
                            'dataset_name': dataset_name,
                            'base_name': base_name,
                            'config': config or 'default',
                            'essay_column_original': dataset_config["essay_column"],
                            'score_column_original': dataset_config["score_column"],
                            'prompt_column_original': dataset_config.get("prompt_column", "prompt"),
                            'essay_column_standardized': 'essay_text',
                            'score_column_standardized': 'score',
                            'prompt_column_standardized': 'prompt',
                            'score_range_min': dataset_config["score_range"][0],
                            'score_range_max': dataset_config["score_range"][1],
                            'huggingface_id': dataset_config["huggingface_id"]
                        })
                        
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
                                
                                # Prepare CSV content with metadata
                                csv_content = []
                                for i, item in enumerate(raw_data):
                                    row = item.get("row", {})
                                    
                                    essay_id = dataset_manager._get_column_value(row, ["essay_id", "id", "sis_id"]) or f"{dataset_name}_{split}_{i}"
                                    essay_text = dataset_manager._get_column_value(row, [dataset_config["essay_column"]])
                                    prompt = dataset_manager._get_column_value(row, [dataset_config.get("prompt_column", "prompt")]) or f"Prompt for {dataset_name}"
                                    score = dataset_manager._get_column_value(row, [dataset_config["score_column"]])
                                    
                                    if essay_text:
                                        csv_row = {
                                            'essay_id': str(essay_id),
                                            'essay_text': str(essay_text),
                                            'prompt': str(prompt),
                                            'dataset_name': dataset_name,
                                            'original_essay_column': dataset_config["essay_column"],
                                            'original_score_column': dataset_config["score_column"],
                                            'original_prompt_column': dataset_config.get("prompt_column", "prompt"),
                                            'config': dataset_config.get("config", "default"),
                                            'score_range_min': dataset_config["score_range"][0],
                                            'score_range_max': dataset_config["score_range"][1]
                                        }
                                        
                                        # Add score column based on split
                                        if split == 'test':
                                            csv_row['score'] = ''
                                            csv_row['score_note'] = 'Empty for prediction task'
                                        else:
                                            csv_row['score'] = score if score is not None else ''
                                            csv_row['score_note'] = 'Human score for training/validation'
                                        
                                        csv_content.append(csv_row)
                                
                                # Write CSV file
                                if csv_content:
                                    filename = f"{split}{file_suffix}.csv"
                                    csv_path = os.path.join(dataset_dir, filename)
                                    
                                    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
                                        fieldnames = [
                                            'essay_id', 'essay_text', 'prompt', 'score', 'score_note',
                                            'dataset_name', 'original_essay_column', 'original_score_column', 
                                            'original_prompt_column', 'config', 'score_range_min', 'score_range_max'
                                        ]
                                        writer = csv.DictWriter(f, fieldnames=fieldnames)
                                        writer.writeheader()
                                        writer.writerows(csv_content)
                                    
                                    total_files_created += 1
                                    print(f"    ✅ Created {filename} with {len(csv_content)} rows + metadata")
                                
                            except Exception as e:
                                print(f"    ❌ Error processing {split}: {e}")
                                continue
                        
                        # Track dataset info for README
                        dataset_info.append({
                            'dataset_name': dataset_name,
                            'base_name': base_name,
                            'config': config,
                            'description': dataset_config['description'],
                            'score_range': dataset_config['score_range'],
                            'huggingface_id': dataset_config['huggingface_id'],
                            'original_columns': {
                                'essay': dataset_config["essay_column"],
                                'score': dataset_config["score_column"],
                                'prompt': dataset_config.get("prompt_column", "prompt")
                            }
                        })
                        
                        print(f"    ✅ Processed {dataset_name}")
                    
                    except Exception as e:
                        print(f"    ❌ Error processing {dataset_name}: {e}")
                        continue
            
            if total_files_created == 0:
                raise HTTPException(status_code=500, detail="No data could be loaded for any dataset")
            
            # Create comprehensive master README
            readme_content = f"""# BESESR Datasets Bundle - Complete Collection with Metadata

Downloaded: {datetime.now().isoformat()}
Total Datasets: {len(dataset_info)}
Total Files: {total_files_created}

## 🎯 Quick Start

### For Researchers:
1. **Training**: Use `train.csv` and `validation.csv` files
2. **Prediction**: Run your model on `test.csv` files (scores are empty)
3. **Submission**: Create CSV with `essay_id,predicted_score` columns
4. **Evaluation**: Upload results via BESESR platform

### File Structure:
This bundle contains datasets organized by base name with configurations:

```
ASAP-AES/                    # Simple dataset
├── train.csv               # Training data WITH scores
├── validation.csv          # Validation data WITH scores  
└── test.csv               # Test data WITHOUT scores

BEEtlE/                     # Multi-config dataset
├── train_2way.csv         # 2-way classification training
├── test_2way.csv          # 2-way classification test
├── train_3way.csv         # 3-way classification training
└── test_3way.csv          # 3-way classification test

grade_like_a_human_dataset_os/  # Question-based dataset
├── q1/
│   ├── train.csv
│   ├── validation.csv
│   └── test.csv
├── q2/
│   ├── train.csv
│   ├── validation.csv
│   └── test.csv
└── ... (q3-q6)
```

## 📊 Column Format

All CSV files use a **standardized format** with **metadata** for transparency:

### Standard Columns:
- `essay_id`: Unique identifier
- `essay_text`: The essay content to score  
- `prompt`: Writing prompt/question
- `score`: Human score (empty in test files)
- `score_note`: Information about score column

### Metadata Columns:
- `dataset_name`: Source dataset
- `original_essay_column`: Original essay column name
- `original_score_column`: Original score column name
- `original_prompt_column`: Original prompt column name
- `config`: Dataset configuration used
- `score_range_min/max`: Valid score range

## 📋 Dataset Information:

"""
            
            # Group dataset info by base name for README
            grouped_info = {}
            for info in dataset_info:
                base = info['base_name']
                if base not in grouped_info:
                    grouped_info[base] = []
                grouped_info[base].append(info)
            
            for base_name, configs in grouped_info.items():
                readme_content += f"""
### {base_name}
- **Configurations**: {len(configs)} ({', '.join([c['config'] or 'default' for c in configs])})
- **HuggingFace ID**: {configs[0]['huggingface_id']}
- **Description**: {configs[0]['description']}
- **Score Range**: {configs[0]['score_range'][0]} - {configs[0]['score_range'][1]}
- **Original Columns**: essay=`{configs[0]['original_columns']['essay']}`, score=`{configs[0]['original_columns']['score']}`
"""
            
            readme_content += f"""

## 🔄 Column Mappings

The standardized format maps original columns as follows:

| Dataset | Essay Column | Score Column | Prompt Column |
|---------|-------------|--------------|---------------|"""

            for mapping in column_mappings[:10]:  # Show first 10 as examples
                readme_content += f"""
| {mapping['dataset_name']} | `{mapping['essay_column_original']}` → `essay_text` | `{mapping['score_column_original']}` → `score` | `{mapping['prompt_column_original']}` → `prompt` |"""

            readme_content += f"""

*See `all_column_mappings.csv` for complete mapping details.*

## 📤 Expected Submission Format

For each dataset, create a results file:

```csv
essay_id,predicted_score
ASAP-AES_test_0,3.5
ASAP-AES_test_1,4.2
BEEtlE_2way_test_0,1
BEEtlE_3way_test_1,2
grade_like_a_human_dataset_os_q1_test_0,85.5
```

## 🎯 Evaluation

- Submit results via BESESR platform
- Automatic evaluation against ground truth
- Leaderboard ranking across all datasets
- Multiple metrics (correlation, accuracy, etc.)

## 📚 Additional Files

- `all_column_mappings.csv`: Complete column mapping reference
- `dataset_summary.json`: Machine-readable dataset information
- Individual README files in each dataset folder

## 🔗 Resources

- **Platform**: BESESR Evaluation Platform
- **Documentation**: Visit platform for detailed guides
- **Support**: Contact via platform interface

---
*This bundle provides standardized access to {len(dataset_info)} essay scoring datasets with full metadata transparency.*
"""

            # Create ZIP file with comprehensive metadata
            zip_buffer = io.BytesIO()
            with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                
                # Add all dataset files with directory structure
                for root, dirs, files in os.walk(temp_dir):
                    for file in files:
                        file_path = os.path.join(root, file)
                        arcname = os.path.relpath(file_path, temp_dir)
                        zip_file.write(file_path, arcname)
                
                # Add master README
                zip_file.writestr("README.md", readme_content)
                
                # Add complete column mappings CSV
                mappings_csv = "dataset_name,base_name,config,original_essay_col,original_score_col,original_prompt_col,standardized_essay_col,standardized_score_col,standardized_prompt_col,score_range_min,score_range_max,huggingface_id\n"
                for mapping in column_mappings:
                    mappings_csv += f"{mapping['dataset_name']},{mapping['base_name']},{mapping['config']},{mapping['essay_column_original']},{mapping['score_column_original']},{mapping['prompt_column_original']},{mapping['essay_column_standardized']},{mapping['score_column_standardized']},{mapping['prompt_column_standardized']},{mapping['score_range_min']},{mapping['score_range_max']},{mapping['huggingface_id']}\n"
                
                zip_file.writestr("all_column_mappings.csv", mappings_csv)
                
                # Add dataset summary JSON
                summary = {
                    "download_date": datetime.now().isoformat(),
                    "total_datasets": len(dataset_info),
                    "total_files": total_files_created,
                    "datasets": dataset_info,
                    "column_mappings": column_mappings
                }
                zip_file.writestr("dataset_summary.json", json.dumps(summary, indent=2))
            
            zip_buffer.seek(0)
            
            print(f"✅ Created comprehensive bundle: {total_files_created} files + metadata across {len(dataset_groups)} datasets")
            
            return Response(
                content=zip_buffer.getvalue(),
                media_type="application/zip",
                headers={"Content-Disposition": "attachment; filename=besesr_datasets_with_metadata.zip"}
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
        print(f"❌ Error creating comprehensive dataset bundle: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to create dataset bundle: {str(e)}")

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