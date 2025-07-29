# app/schemas/output_submissions.py

from pydantic import BaseModel

class OutputSubmissionResponse(BaseModel):
    submission_id: int
    message: str

    class Config:
        orm_mode = True