# app/models/database.py

from sqlalchemy import Column, Integer, String, DateTime, Float, Text, Boolean, JSON, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from datetime import datetime, timezone

Base = declarative_base()

class Dataset(Base):
    """Database table for storing dataset information"""
    __tablename__ = "datasets"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), unique=True, index=True, nullable=False)
    description = Column(Text)
    huggingface_id = Column(String(255))
    essay_count = Column(Integer, default=0)
    avg_essay_length = Column(Float, default=0.0)
    score_range_min = Column(Float, default=0.0)
    score_range_max = Column(Float, default=6.0)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))


class BenchmarkSession(Base):
    """Database table for tracking complete 25-dataset benchmark submissions"""
    __tablename__ = "benchmark_sessions"
    
    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String(255), unique=True, index=True, nullable=False)
    model_name = Column(String(255), nullable=False)
    submitter_name = Column(String(255), nullable=False)
    submitter_email = Column(String(255), nullable=False)
    model_description = Column(Text)
    paper_url = Column(String(512))
    
    # Progress tracking
    status = Column(String(50), default="active")  # "active", "completed", "failed"
    datasets_completed = Column(Integer, default=0)
    total_datasets = Column(Integer, default=25)  # Updated to 25 datasets
    
    # Results
    average_score = Column(Float)
    benchmark_results = Column(JSON)  # Store complete results
    
    # Timestamps
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))
    completed_at = Column(DateTime)
    error_message = Column(Text)
    
    # Relationships
    submissions = relationship("OutputSubmission", back_populates="benchmark_session")


class OutputSubmission(Base):
    """Database table for storing individual dataset submissions"""
    __tablename__ = "output_submissions"
    
    id = Column(Integer, primary_key=True, index=True)
    submitter_name = Column(String(255), nullable=False, index=True)
    dataset_name = Column(String(255), nullable=False, index=True)
    status = Column(String(50), default="submitted", index=True)  
    submitter_email = Column(String(255), nullable=False)
    file_path = Column(String(512), nullable=False)
    file_format = Column(String(10), default="csv")
    description = Column(Text)
    evaluation_result = Column(JSON)
    error_message = Column(Text)
    submission_time = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    processing_time = Column(Float)
    
    # Foreign key to benchmark session (optional)
    benchmark_session_id = Column(String(255), ForeignKey("benchmark_sessions.session_id"))
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))
    
    # Relationships
    evaluations = relationship("EvaluationResult", back_populates="submission", cascade="all, delete-orphan")
    benchmark_session = relationship("BenchmarkSession", back_populates="submissions")


class EvaluationResult(Base):
    """Database table for storing evaluation metrics for each submission"""
    __tablename__ = "evaluation_results"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    submission_id = Column(Integer, ForeignKey("output_submissions.id"), nullable=False)
    dataset_name = Column(String(255), nullable=False)
    
    # Core metrics (your 5 evaluation metrics)
    quadratic_weighted_kappa = Column(Float, nullable=True)
    pearson_correlation = Column(Float, nullable=True)
    spearman_correlation = Column(Float, nullable=True)
    mean_absolute_error = Column(Float, nullable=True)
    root_mean_squared_error = Column(Float, nullable=True)
    f1_score = Column(Float, nullable=True)
    accuracy = Column(Float, nullable=True)
    
    # Evaluation metadata
    essays_evaluated = Column(Integer, default=0)
    evaluation_time = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    evaluation_duration = Column(Float, nullable=True)
    status = Column(String(50), default="completed")
    error_message = Column(Text, nullable=True)
    detailed_metrics = Column(JSON, nullable=True)
    
    # Relationship
    submission = relationship("OutputSubmission", back_populates="evaluations")


class Essay(Base):
    """Database table for storing individual essays and their scores"""
    __tablename__ = "essays"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    essay_id = Column(String(255), unique=True, nullable=False, index=True)
    dataset_name = Column(String(255), nullable=False, index=True)
    essay_text = Column(Text, nullable=False)
    prompt = Column(Text, nullable=True)
    
    # Human scores (reference scores for evaluation)
    holistic_score = Column(Float, nullable=True)
    content_score = Column(Float, nullable=True)
    organization_score = Column(Float, nullable=True)
    style_score = Column(Float, nullable=True)
    grammar_score = Column(Float, nullable=True)
    
    # Essay metadata
    word_count = Column(Integer, nullable=True)
    sentence_count = Column(Integer, nullable=True)
    paragraph_count = Column(Integer, nullable=True)
    grade_level = Column(String(50), nullable=True)
    essay_attributes = Column(JSON, nullable=True)
    created_time = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    # Add indexes for better performance
    __table_args__ = (
        {'mysql_engine': 'InnoDB'},
    )