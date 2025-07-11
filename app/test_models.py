from app.api.models.essay import EssaySubmission
from app.api.models.grading import GradingResponse
from datetime import datetime

# Test EssaySubmission
submission = EssaySubmission(
    essay_text="Test essay",
    dataset_name="ASAP-AES"
)
print("EssaySubmission works:", submission)

# Test GradingResponse
response = GradingResponse(
    scores={"holistic_score": 4.5},
    feedback="Test feedback",
    dataset_used="ASAP-AES",
    model_used="test_model",
    processing_time=0.5,
    timestamp=datetime.now()
)
print("GradingResponse works:", response)