from fastapi import APIRouter, HTTPException
from app.api.models.essay import EssaySubmission
from app.api.models.grading import GradingResponse, EssayGradingRequest, EssayGradingResponse
from datetime import datetime
from app.services.dataset_loader import dataset_manager

router = APIRouter(prefix="/essays", tags=["essays - internal testing"])

@router.post("/grade", response_model=GradingResponse)
async def grade_essay_internal(submission: EssaySubmission):
    """Internal endpoint for testing essay grading (not part of main benchmarking)"""
    
    # Validate dataset exists
    dataset_config = dataset_manager.datasets_config.get(submission.dataset_name)
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
router = APIRouter()

@router.get("/test-load/{dataset_name}")
async def test_load_dataset(dataset_name: str, sample_size: int = 5):
    """Test loading sample essays from Hugging Face dataset"""
    result = dataset_manager.load_dataset_for_evaluation(dataset_name, sample_size)
    return {
        "num_loaded": len(result),
        "sample": result[:1]  # show first essay only
    }