# app/models/database.py

from sqlalchemy import Column, Integer, String, DateTime, Float, Text, Boolean, JSON, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from datetime import datetime
from pydantic import BaseModel, ConfigDict
from typing import Optional, Dict, Any, List

Base = declarative_base()

# ============================================================================
# SQLAlchemy Database Models
# ============================================================================

class Dataset(Base):
    __tablename__ = "datasets"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True, index=True, nullable=False)
    description = Column(Text)
    huggingface_id = Column(String)
    essay_count = Column(Integer, default=0)
    avg_essay_length = Column(Float, default=0.0)
    score_range_min = Column(Float, default=0.0)
    score_range_max = Column(Float, default=6.0)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow)


class OutputSubmission(Base):
    __tablename__ = "output_submissions"
    
    id = Column(Integer, primary_key=True, index=True)
    dataset_name = Column(String, nullable=False)
    submitter_name = Column(String, nullable=False)
    submitter_email = Column(String, nullable=False)
    file_path = Column(String, nullable=False)
    file_format = Column(String, default="csv")
    status = Column(String, default="submitted")
    description = Column(Text)
    evaluation_result = Column(JSON)
    error_message = Column(Text)
    submission_time = Column(DateTime, default=datetime.utcnow)
    processing_time = Column(Float)
    
    # Progressive upload fields
    benchmark_session_id = Column(String, ForeignKey("benchmark_sessions.session_id"))
    updated_at = Column(DateTime, default=datetime.utcnow)
    
    evaluations = relationship("EvaluationResult", back_populates="submission")
    benchmark_session = relationship("BenchmarkSession", back_populates="submissions")

class EvaluationResult(Base):
    __tablename__ = "evaluation_results"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    submission_id = Column(Integer, ForeignKey("output_submissions.id"), nullable=False)
    dataset_name = Column(String, nullable=False)
    
    # Core metrics
    quadratic_weighted_kappa = Column(Float, nullable=True)
    pearson_correlation = Column(Float, nullable=True)
    spearman_correlation = Column(Float, nullable=True)
    mean_absolute_error = Column(Float, nullable=True)
    root_mean_squared_error = Column(Float, nullable=True)
    f1_score = Column(Float, nullable=True)
    accuracy = Column(Float, nullable=True)
    
    # Evaluation metadata
    essays_evaluated = Column(Integer, default=0)
    evaluation_time = Column(DateTime, default=datetime.utcnow)
    evaluation_duration = Column(Float, nullable=True)
    status = Column(String, default="completed")
    error_message = Column(Text, nullable=True)
    detailed_metrics = Column(JSON, nullable=True)
    
    # Relationship
    submission = relationship("OutputSubmission", back_populates="evaluations")


class Essay(Base):
    __tablename__ = "essays"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    essay_id = Column(String, unique=True, nullable=False)
    dataset_name = Column(String, nullable=False)
    essay_text = Column(Text, nullable=False)
    prompt = Column(Text, nullable=True)
    
    # Human scores
    holistic_score = Column(Float, nullable=True)
    content_score = Column(Float, nullable=True)
    organization_score = Column(Float, nullable=True)
    style_score = Column(Float, nullable=True)
    grammar_score = Column(Float, nullable=True)
    
    # Essay metadata
    word_count = Column(Integer, nullable=True)
    sentence_count = Column(Integer, nullable=True)
    paragraph_count = Column(Integer, nullable=True)
    grade_level = Column(String, nullable=True)
    essay_attributes = Column(JSON, nullable=True)
    created_time = Column(DateTime, default=datetime.utcnow)

# ============================================================================
# Pydantic Models (for API validation) - Fixed protected namespace issues
# ============================================================================

class DatasetCreate(BaseModel):
    model_config = ConfigDict(protected_namespaces=())  # Fix for protected namespace warnings
    
    name: str
    description: str
    huggingface_id: Optional[str] = None
    essay_count: int = 0
    avg_essay_length: float = 0.0
    score_range_min: float = 0.0
    score_range_max: float = 6.0
    is_active: bool = True

class DatasetResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    
    id: int
    name: str
    description: str
    huggingface_id: Optional[str]
    essay_count: int
    avg_essay_length: float
    score_range_min: float
    score_range_max: float
    is_active: bool
    created_at: datetime

class OutputSubmissionCreate(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    
    dataset_name: str
    submitter_name: str
    submitter_email: str
    file_path: str
    file_format: str = "csv"
    description: Optional[str] = None

class OutputSubmissionResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    
    id: int
    dataset_name: str
    submitter_name: str
    submitter_email: str
    status: str
    submission_time: datetime
    evaluation_result: Optional[Dict[str, Any]] = None

class EvaluationResultCreate(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    
    submission_id: int
    dataset_name: str
    quadratic_weighted_kappa: float = 0.0
    pearson_correlation: float = 0.0
    spearman_correlation: float = 0.0
    mean_absolute_error: float = 999.0
    root_mean_squared_error: float = 999.0
    f1_score: float = 0.0
    accuracy: float = 0.0
    essays_evaluated: int = 0
    evaluation_duration: float = 0.0
    detailed_metrics: Optional[Dict[str, Any]] = None

class EvaluationResultResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    
    id: int
    submission_id: int
    dataset_name: str
    quadratic_weighted_kappa: float
    pearson_correlation: float
    spearman_correlation: float
    mean_absolute_error: float
    root_mean_squared_error: float
    f1_score: float
    accuracy: float
    essays_evaluated: int
    status: str
    evaluation_time: datetime

class EssayCreate(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    
    essay_id: str
    dataset_name: str
    essay_text: str
    prompt: Optional[str] = None
    holistic_score: Optional[float] = None
    content_score: Optional[float] = None
    organization_score: Optional[float] = None
    style_score: Optional[float] = None
    grammar_score: Optional[float] = None
    word_count: Optional[int] = None
    sentence_count: Optional[int] = None
    paragraph_count: Optional[int] = None
    grade_level: Optional[str] = None
    essay_attributes: Optional[Dict[str, Any]] = None

class EssayResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    
    id: int
    essay_id: str
    dataset_name: str
    essay_text: str
    prompt: Optional[str]
    holistic_score: Optional[float]
    content_score: Optional[float]
    organization_score: Optional[float]
    style_score: Optional[float]
    grammar_score: Optional[float]
    word_count: Optional[int]
    sentence_count: Optional[int]
    paragraph_count: Optional[int]
    grade_level: Optional[str]
    essay_attributes: Optional[Dict[str, Any]]
    created_time: datetime

# ============================================================================
# Additional Response Models
# ============================================================================

class LeaderboardEntry(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    
    rank: int
    submission_id: int
    submitter_name: str
    dataset_name: str
    quadratic_weighted_kappa: float
    pearson_correlation: float
    mean_absolute_error: float
    accuracy: float
    essays_evaluated: int
    submission_time: Optional[str] = None
    submission_type: str = "csv_upload"

class PlatformStats(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    
    total_submissions: int
    total_evaluations: int
    total_datasets: int
    completed_evaluations: int
    success_rate: float

class HealthCheck(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    
    status: str
    timestamp: str
    database: str
    error: Optional[str] = None


class BenchmarkSession(Base):
    __tablename__ = "benchmark_sessions"
    
    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String, unique=True, index=True, nullable=False)
    model_name = Column(String, nullable=False)
    submitter_name = Column(String, nullable=False)
    submitter_email = Column(String, nullable=False)
    model_description = Column(Text)
    paper_url = Column(String)
    
    # Progress tracking
    status = Column(String, default="active")
    datasets_completed = Column(Integer, default=0)
    total_datasets = Column(Integer, default=12)
    
    # Results
    average_score = Column(Float)
    benchmark_results = Column(Text)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.now)
    updated_at = Column(DateTime, default=datetime.now)
    completed_at = Column(DateTime)
    error_message = Column(Text)
    
    # Relationships
    submissions = relationship("OutputSubmission", back_populates="benchmark_session")