from typing import Dict, List, Optional
from app.api.models.essay import DatasetConfig

# Dataset configurations
DATASETS = {
    # Datasets with rubrics
    "ASAP-AES": DatasetConfig(
    name="ASAP-AES",
    description="ASAP Automated Essay Scoring dataset - 8 essay sets with different scoring ranges",
    huggingface_id="nlpatunt/ASAP-AES",
    evaluation_metrics=["resolved_score", "domain1_score", "domain2_score", "rubric_score", "rater1_score", "rater2_score", "rater3_score"],
    score_ranges={
        "essay_set_1": {"resolved_score": (2, 12), "rubric_score": (1, 6)},
        "essay_set_2": {"domain1_score": (1, 6), "domain2_score": (1, 4), "domain1_rubric": (1, 6), "domain2_rubric": (1, 4)},
        "essay_set_3": {"resolved_score": (0, 3), "rubric_score": (0, 3)},
        "essay_set_4": {"resolved_score": (0, 3), "rubric_score": (0, 3)},
        "essay_set_5": {"resolved_score": (0, 4), "rubric_score": (0, 4)},
        "essay_set_6": {"resolved_score": (0, 4), "rubric_score": (0, 4)},
        "essay_set_7": {"resolved_score": (0, 30), "rubric_score": (0, 15)},
        "essay_set_8": {"resolved_score": (0, 60), "rater1_score": (0, 30), "rater2_score": (0, 30), "rater3_score": (0, 60)}
        }
    ),
    "ASAP-SAS": DatasetConfig(
        name="ASAP-SAS",
        description="ASAP Short Answer Scoring dataset",
        huggingface_id=None,  # update with actual ID
        evaluation_metrics=["holistic_score"],
        score_ranges={"holistic_score": (0.0, 3.0)}
    ),
    "rice_chem": DatasetConfig(
        name="rice_chem",
        description="Rice Chemistry dataset",
        huggingface_id=None,  # update with actual ID
        evaluation_metrics=["content_accuracy", "scientific_reasoning"],
        score_ranges={"content_accuracy": (0.0, 5.0), "scientific_reasoning": (0.0, 5.0)}
    ),
    "CSEE": DatasetConfig(
        name="CSEE",
        description="Computer Science Essay Evaluation dataset",
        huggingface_id=None,  # update with actual ID
        evaluation_metrics=["technical_accuracy", "explanation_quality"],
        score_ranges={"technical_accuracy": (0.0, 5.0), "explanation_quality": (0.0, 5.0)}
    ),
    "EFL": DatasetConfig(
        name="EFL",
        description="English as Foreign Language dataset",
        huggingface_id=None,  # update with actual ID
        evaluation_metrics=["grammar", "vocabulary", "fluency"],
        score_ranges={"grammar": (0.0, 5.0), "vocabulary": (0.0, 5.0), "fluency": (0.0, 5.0)}
    ),
    "grade_like_a_human_dataset_os": DatasetConfig(
        name="grade_like_a_human_dataset_os",
        description="Grade Like a Human Open Source dataset",
        huggingface_id=None,  # update with actual ID
        evaluation_metrics=["overall_quality", "rubric_adherence"],
        score_ranges={"overall_quality": (0.0, 100.0), "rubric_adherence": (0.0, 5.0)}
    ),
    "persuade_2": DatasetConfig(
        name="persuade_2",
        description="Persuasive Essay dataset version 2",
        huggingface_id=None,  # update with actual ID
        evaluation_metrics=["argument_strength", "evidence_quality", "organization"],
        score_ranges={"argument_strength": (0.0, 6.0), "evidence_quality": (0.0, 6.0), "organization": (0.0, 6.0)}
    ),
    
    # Datasets without rubrics
    "ASAP2": DatasetConfig(
        name="ASAP2",
        description="ASAP version 2 dataset",
        huggingface_id=None,  # update with actual ID
        evaluation_metrics=["holistic_score"],
        score_ranges={"holistic_score": (0.0, 6.0)}
    ),
    "ASAP_plus_plus": DatasetConfig(
        name="ASAP_plus_plus",
        description="ASAP Plus Plus enhanced dataset",
        huggingface_id=None,  # update with actual ID
        evaluation_metrics=["holistic_score"],
        score_ranges={"holistic_score": (0.0, 6.0)}
    ),
    "SciEntSBank": DatasetConfig(
        name="SciEntSBank",
        description="Science Entailment Bank dataset",
        huggingface_id="nlpatunt/SciEntSBank",
        evaluation_metrics=["semantic_similarity", "factual_accuracy"],
        score_ranges={"semantic_similarity": (0.0, 1.0), "factual_accuracy": (0.0, 1.0)}
    ),
    "BEEtlE": DatasetConfig(
        name="BEEtlE",
        description="BEEtlE dataset for educational assessment",
        huggingface_id=None,  # update with actual ID
        evaluation_metrics=["correctness", "completeness"],
        score_ranges={"correctness": (0.0, 1.0), "completeness": (0.0, 1.0)}
    ),
    "automatic_short_answer_grading": DatasetConfig(
        name="automatic_short_answer_grading",
        description="Automatic Short Answer Grading dataset",
        huggingface_id=None,  # update with actual ID
        evaluation_metrics=["accuracy", "relevance"],
        score_ranges={"accuracy": (0.0, 5.0), "relevance": (0.0, 5.0)}
    )
}

def get_dataset_config(dataset_name: str) -> Optional[DatasetConfig]:
    return DATASETS.get(dataset_name)

def get_all_datasets() -> Dict[str, DatasetConfig]:
    return DATASETS