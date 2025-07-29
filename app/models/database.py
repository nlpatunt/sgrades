from sqlalchemy import Column, Integer, String, DateTime, Float, Text, Boolean, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from datetime import datetime

Base = declarative_base()

class OutputSubmission(Base):
    __tablename__ = "output_submissions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    dataset_name = Column(String, nullable=False)
    submitter_name = Column(String, nullable=False)
    submitter_email = Column(String, nullable=False)
    submission_filename = Column(String, nullable=False)
    file_format = Column(String, default="csv")  # 'csv' or 'json'
    submission_time = Column(DateTime, default=datetime.utcnow)
    status = Column(String, default="submitted")  # submitted, evaluating, completed, failed
    evaluation_result = Column(JSON, nullable=True)
    error_message = Column(Text, nullable=True)

    def to_dict(self):
        return {
            'id': self.id,
            'dataset_name': self.dataset_name,
            'submitter_name': self.submitter_name,
            'submitter_email': self.submitter_email,
            'submission_filename': self.submission_filename,
            'file_format': self.file_format,
            'submission_time': self.submission_time.isoformat() if self.submission_time else None,
            'status': self.status,
            'evaluation_result': self.evaluation_result,
            'error_message': self.error_message
        }


class Dataset(Base):
    __tablename__ = "datasets"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String, unique=True, nullable=False)
    description = Column(Text, nullable=False)
    huggingface_id = Column(String, nullable=True)
    essay_count = Column(Integer, default=0)
    avg_essay_length = Column(Float, nullable=True)
    score_range_min = Column(Float, default=1.0)
    score_range_max = Column(Float, default=5.0)
    created_time = Column(DateTime, default=datetime.utcnow)
    is_active = Column(Boolean, default=True)

class EvaluationResult(Base):
    __tablename__ = "evaluation_results"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    submission_id = Column(Integer, nullable=False)  # FK to OutputSubmission.id
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
    evaluation_duration = Column(Float, nullable=True)  # seconds
    status = Column(String, default="completed")  # completed, failed, partial
    error_message = Column(Text, nullable=True)
    
    # Additional metrics storage
    detailed_metrics = Column(JSON, nullable=True)
    
    def to_dict(self):
        return {
            'id': self.id,
            'submission_id': self.submission_id,
            'dataset_name': self.dataset_name,
            'quadratic_weighted_kappa': self.quadratic_weighted_kappa,
            'pearson_correlation': self.pearson_correlation,
            'spearman_correlation': self.spearman_correlation,
            'mean_absolute_error': self.mean_absolute_error,
            'root_mean_squared_error': self.root_mean_squared_error,
            'f1_score': self.f1_score,
            'accuracy': self.accuracy,
            'essays_evaluated': self.essays_evaluated,
            'evaluation_time': self.evaluation_time.isoformat() if self.evaluation_time else None,
            'evaluation_duration': self.evaluation_duration,
            'status': self.status,
            'error_message': self.error_message,
            'detailed_metrics': self.detailed_metrics
        }

class Essay(Base):
    __tablename__ = "essays"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    essay_id = Column(String, unique=True, nullable=False)
    dataset_name = Column(String, nullable=False)
    essay_text = Column(Text, nullable=False)
    prompt = Column(Text, nullable=True)
    
    # Human scores (gold standard)
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
    
    # Additional essay attributes
    essay_attributes = Column(JSON, nullable=True)
    
    created_time = Column(DateTime, default=datetime.utcnow)
    
    def to_dict(self):
        return {
            'id': self.id,
            'essay_id': self.essay_id,
            'dataset_name': self.dataset_name,
            'essay_text': self.essay_text,
            'prompt': self.prompt,
            'holistic_score': self.holistic_score,
            'content_score': self.content_score,
            'organization_score': self.organization_score,
            'style_score': self.style_score,
            'grammar_score': self.grammar_score,
            'word_count': self.word_count,
            'sentence_count': self.sentence_count,
            'paragraph_count': self.paragraph_count,
            'grade_level': self.grade_level,
            'essay_attributes': self.essay_attributes,
            'created_time': self.created_time.isoformat() if self.created_time else None
        }