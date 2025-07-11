from fastapi import APIRouter, HTTPException
from typing import List, Dict, Any
from app.config.datasets import get_all_datasets, get_dataset_config

router = APIRouter(prefix="/datasets", tags=["datasets"])

@router.get("/")
async def list_datasets():
    """Get list of all available datasets"""
    datasets = get_all_datasets()
    return {
        "total_datasets": len(datasets),
        "datasets": [
            {
                "name": name,
                "description": config.description,
                "huggingface_id": config.huggingface_id
            }
            for name, config in datasets.items()
        ]
    }

@router.get("/{dataset_name}/info")
async def get_dataset_info(dataset_name: str):
    """Get detailed information about a specific dataset"""
    config = get_dataset_config(dataset_name)
    if not config:
        raise HTTPException(status_code=404, detail="Dataset not found")
    return config