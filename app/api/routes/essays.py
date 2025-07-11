from fastapi import APIRouter, HTTPException
from app.api.models.essay import EssaySubmission
from app.api.models.grading import GradingResponse, EssayGradingRequest, EssayGradingResponse
from app.config.datasets import get_dataset_config
from datetime import datetime

router = APIRouter(prefix="/essays", tags=["essays - internal testing"])

@router.post("/grade", response_model=GradingResponse)
async def grade_essay_internal(submission: EssaySubmission):
    """Internal endpoint for testing essay grading (not part of main benchmarking)"""
    
    # Validate dataset exists
    dataset_config = get_dataset_config(submission.dataset_name)
    if not dataset_config:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    # Mock response for internal testing
    return GradingResponse(
        scores={"holistic_score": 4.5},
        feedback="This is an internal testing endpoint. Use /models/submit for benchmarking.",
        dataset_used=submission.dataset_name,
        model_used="internal_test",
        processing_time=0.1,
        timestamp=datetime.now()
    )

@router.post("/test-model-api", response_model=EssayGradingResponse)
async def test_model_api_format(request: EssayGradingRequest):
    """Test endpoint to check the expected API format for submitted models"""
    
    # This shows researchers what format their model API should return
    return EssayGradingResponse(
        essay_id=request.essay_id,
        scores={"holistic_score": 4.2, "content": 4.0, "organization": 4.5},
        processing_time=1.2,
        model_confidence=0.85,
        additional_info={"model_version": "1.0", "notes": "Example response format"}
    )