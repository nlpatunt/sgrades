from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Union
from datetime import datetime

class DatasetInfo(BaseModel):
    name: str
    description: str
    huggingface_id: str
    essay_column: str
    score_column: str
    prompt_column: str
    score_range: List[Union[int, float]]
    split: str
    status: str = "active"

class DatasetConfiguration(BaseModel):
    essay_column: str
    score_column: str
    prompt_column: str
    score_range: List[Union[int, float]]
    split: str

class DatasetDetails(BaseModel):
    name: str
    description: str
    huggingface_id: str
    configuration: DatasetConfiguration
    huggingface_info: Optional[Dict[str, Any]] = None
    sample_available: bool = True
    status: str = "active"

class EssayPreview(BaseModel):
    essay_id: str
    essay_preview: str
    prompt: str
    human_score: Union[int, float]
    score_range: List[Union[int, float]]
    word_count: int
    metadata: Dict[str, Any] = {}

class DatasetSample(BaseModel):
    dataset_name: str
    sample_size: int
    requested_size: int
    essays: List[EssayPreview]
    loaded_at: str

class DatasetsListResponse(BaseModel):
    datasets: List[DatasetInfo]
    total_count: int
    data_source: str = "dynamic_huggingface"
    last_updated: str

class DatasetHealthCheck(BaseModel):
    status: str
    total_datasets_configured: int
    authentication: str
    test_dataset: Optional[str] = None
    test_sample_loaded: bool = False
    timestamp: str

class BenchmarkSubmissionRequest(BaseModel):
    submitter_name: str = Field(..., min_length=2, max_length=100)
    submitter_email: str = Field(..., pattern=r'^[^@]+@[^@]+\.[^@]+$')
    model_name: str = Field(..., min_length=2, max_length=100)
    model_description: Optional[str] = Field(None, max_length=1000)

class SingleTestRequest(BaseModel):
    model_name: str = Field(..., min_length=2, max_length=100)
    dataset_name: str
    submitter_name: str = Field(..., min_length=2, max_length=100)
    submitter_email: str = Field(..., pattern=r'^[^@]+@[^@]+\.[^@]+$')
    model_description: Optional[str] = Field(None, max_length=500)

class EvaluationMetrics(BaseModel):
    quadratic_weighted_kappa: float
    pearson_correlation: float
    spearman_correlation: float
    mean_absolute_error: float
    root_mean_squared_error: float
    f1_score: float
    accuracy: float
    essays_evaluated: int
    match_rate: float

class BenchmarkSubmissionResponse(BaseModel):
    message: str
    submitter_name: str
    model_name: str
    datasets_processed: int
    failed_datasets: List[str]
    total_essays_evaluated: int
    submission_ids: List[int]
    avg_quadratic_weighted_kappa: float
    avg_pearson_correlation: float
    avg_mean_absolute_error: float
    avg_f1_score: float
    avg_accuracy: float
    status: str
    benchmark_type: str = "complete_15_dataset"
    completion_rate: float

class SingleTestResponse(BaseModel):
    message: str
    note: str
    submission_id: int
    model_name: str
    dataset_name: str
    evaluation_results: EvaluationMetrics
    status: str
    submission_type: str = "individual_test"

class CSVValidationResponse(BaseModel):
    valid: bool
    message: Optional[str] = None
    error: Optional[str] = None
    row_count: Optional[int] = None
    columns: Optional[List[str]] = None
    sample_data: Optional[List[Dict[str, Any]]] = None
    required_columns: Optional[List[str]] = None
    columns_found: Optional[List[str]] = None

class SubmissionTemplate(BaseModel):
    csv_format: Dict[str, Any]
    available_datasets: List[str]
    upload_instructions: Dict[str, Any]
    requirements: Dict[str, Any]
    example_csv: str

class SubmissionStatus(BaseModel):
    submission_id: int
    dataset_name: str
    submitter_name: str
    status: str
    submitted_at: Optional[datetime] = None
    evaluation_result: Optional[Dict[str, Any]] = None
    detailed_evaluations: Optional[List[Dict[str, Any]]] = None
    error_message: Optional[str] = None

class RecentSubmissions(BaseModel):
    submissions: List[Dict[str, Any]]
    total_count: int
    type_filter: str = "all"

class LeaderboardEntry(BaseModel):
    rank: int
    submitter_name: str
    submitter_email: Optional[str] = None
    model_name: Optional[str] = None
    quadratic_weighted_kappa: float
    pearson_correlation: float
    mean_absolute_error: float
    accuracy: float
    essays_evaluated: int
    upload_timestamp: Optional[str] = None
    submission_type: str

class CompleteLeaderboardEntry(BaseModel):
    rank: int
    submitter_name: str
    submitter_email: Optional[str] = None
    total_datasets: int
    total_essays_evaluated: int
    avg_quadratic_weighted_kappa: float
    avg_pearson_correlation: float
    avg_mean_absolute_error: float
    avg_f1_score: float
    avg_accuracy: float
    datasets_completed: List[str]
    completion_rate: float
    benchmark_type: str = "complete_15_dataset"

class ResearcherProgress(BaseModel):
    submitter_name: str
    completed_datasets: int
    total_datasets: int = 15
    completion_percentage: float
    is_complete: bool
    completed_dataset_names: List[str]
    remaining_datasets: int

class PlatformStats(BaseModel):
    total_complete_benchmarks: int
    total_individual_submissions: int
    total_datasets: int = 15
    total_models_submitted: int
    total_evaluations_completed: int
    researchers_with_complete_benchmarks: int
    researchers_in_progress: int
    top_performer: Optional[str] = None
    latest_submission: Optional[str] = None
    avg_evaluation_time: str = "15-30 minutes"
    datasets_available: List[str]
    platform_uptime_days: int
    database_status: str = "connected"
    api_version: str = "1.0.0"
    evaluation_success_rate: float = 0.95
    benchmark_type: str = "complete_15_dataset_only"

class MetricInfo(BaseModel):
    name: str
    display_name: str
    description: str
    range: List[Union[str, float]]
    higher_is_better: bool
    default: bool = False

class AvailableMetrics(BaseModel):
    primary_metrics: List[MetricInfo]
    secondary_metrics: List[MetricInfo]
    supported_sorting: List[str]

class HealthCheck(BaseModel):
    status: str
    service: str
    database_connection: str
    timestamp: str
    complete_benchmarks_available: Optional[int] = None
    models_available: Optional[int] = None
    evaluations_completed: Optional[int] = None
    error: Optional[str] = None

class ErrorResponse(BaseModel):
    error: str
    detail: Optional[str] = None
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())

class NotFoundResponse(BaseModel):
    error: str
    available_options: Optional[List[str]] = None

class DatasetFormatResponse(BaseModel):
    dataset_name: str
    required_columns: List[str]
    score_column: str
    id_column: str
    matching_method: str

class AvailableDatasetsResponse(BaseModel):
    datasets: List[str]
    total: int

class SubmissionResponse(BaseModel):
    success: bool
    evaluation: Optional[Dict[str, Any]] = None
    database: Optional[Dict[str, Any]] = None
    dataset: str
    model_name: Optional[str] = None
    filename: str
    encoding_used: Optional[str] = None
    validation_errors: Optional[List[str]] = None
    validation_warnings: Optional[List[str]] = None
    testing_mode: Optional[bool] = None
    note: Optional[str] = None
    error: Optional[str] = None

class TestSubmissionResponse(BaseModel):
    success: bool
    testing_mode: bool
    dataset: str
    filename: str
    evaluation: Optional[Dict[str, Any]] = None
    metrics: Optional[Dict[str, float]] = None
    encoding_used: Optional[str] = None
    validation_warnings: Optional[List[str]] = None
    validation_errors: Optional[List[str]] = None
    evaluation_error: Optional[str] = None
    note: Optional[str] = None
    error: Optional[str] = None

class SubmissionsStatsResponse(BaseModel):
    total_submissions: int
    total_researchers: int
    complete_benchmarks: int
    available_datasets: int
    evaluation_type: str
    matching_method: str

class BatchSubmissionResponse(BaseModel):
    success: bool
    total_files: int
    successful_uploads: int
    failed_uploads: int
    model_name: str
    results: List[Dict[str, Any]]
    note: str

class ApiInfoResponse(BaseModel):
    workflow: List[str]
    version: str
    supported_datasets: int
    documentation: str