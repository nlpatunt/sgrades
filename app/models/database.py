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


class OutputSubmission(Base):
    __tablename__ = "output_submissions"
    
    id = Column(Integer, primary_key=True, index=True)
    dataset_name = Column(String, nullable=False)
    submitter_name = Column(String, nullable=False)
    submitter_email = Column(String, nullable=False)

    # File storage fields
    original_filename = Column(String, nullable=True)
    stored_file_path = Column(String, nullable=True)
    file_hash = Column(String, nullable=True)
    file_size = Column(Integer, nullable=True)
    
    # Metadata
    upload_timestamp = Column(DateTime, default=datetime.utcnow)
    ip_address = Column(String)
    user_agent = Column(String)
    
    # Evaluation results
    evaluation_result = Column(Text)
    status = Column(String, default="pending")
    description = Column(String, nullable=True)
    
    # Audit fields
    is_archived = Column(Boolean, default=False)
    archive_reason = Column(String)
    
    # Foreign key for benchmark session
    benchmark_session_id = Column(Integer, ForeignKey("benchmark_sessions.id"), nullable=True)
    
    # Relationships
    benchmark_session = relationship("BenchmarkSession", back_populates="submissions")
    evaluations = relationship("EvaluationResult", back_populates="submission")


class BenchmarkSession(Base):
    """Database table for tracking complete 15-dataset benchmark submissions"""
    __tablename__ = "benchmark_sessions"
    
    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String(255), unique=True, index=True, nullable=False)
    model_name = Column(String(255), nullable=False)
    submitter_name = Column(String(255), nullable=False)
    submitter_email = Column(String(255), nullable=False)
    model_description = Column(Text)
    paper_url = Column(String(512))
    
    # Progress tracking
    status = Column(String(50), default="active")
    datasets_completed = Column(Integer, default=0)
    total_datasets = Column(Integer, default=24)
    
    # Results
    average_score = Column(Float)
    benchmark_results = Column(JSON)
    
    # Timestamps
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))
    completed_at = Column(DateTime)
    error_message = Column(Text)
    
    # Relationships
    submissions = relationship("OutputSubmission", back_populates="benchmark_session")


class EvaluationResult(Base):
    """Database table for storing evaluation metrics for each submission"""
    __tablename__ = "evaluation_results"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    submission_id = Column(Integer, ForeignKey("output_submissions.id"), nullable=False)
    dataset_name = Column(String(255), nullable=False)
    
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
    evaluation_time = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    evaluation_duration = Column(Float, nullable=True)
    status = Column(String(50), default="completed")
    error_message = Column(Text, nullable=True)
    detailed_metrics = Column(JSON, nullable=True)
    
    # Relationship
    submission = relationship("OutputSubmission", back_populates="evaluations")


class Essay(Base):
    __tablename__ = "essays"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    essay_id = Column(String(255), unique=True, nullable=False, index=True)
    dataset_name = Column(String(255), nullable=False, index=True)
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
    grade_level = Column(String(50), nullable=True)
    essay_attributes = Column(JSON, nullable=True)
    created_time = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    # Add indexes for better performance
    __table_args__ = (
        {'mysql_engine': 'InnoDB'},
    )