from typing import Dict, List, Optional
from app.api.models.essay import DatasetConfig

# Dataset configurations
DATASETS = {
    # Datasets with rubrics
    "ASAP-AES": DatasetConfig(
        name="ASAP-AES",
        description="ASAP Automated Essay Scoring dataset - 8 essay sets with different scoring ranges",
        huggingface_id="nlpatunt/ASAP-AES",
        evaluation_metrics=["inter_rater_agreement", "quadratic_weighted_kappa", "accuracy", "SMD", "p-values"],
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
        description="ASAP Short Answer Scoring dataset - 10 sets of Source Dependent Responses",
        huggingface_id="nlpatunt/ASAP-SAS",
        evaluation_metrics=["quadratic_weighted_kappa", "accuracy"],
        score_ranges={
            "essay_set_1": {"score1": (0, 3), "score2": (0, 3), "rubric_score": (0, 3), "has_reading_passage": False},
            "essay_set_2": {"score1": (0, 3), "score2": (0, 3), "rubric_score": (0, 3), "has_reading_passage": False},
            "essay_set_3": {"score1": (0, 2), "score2": (0, 2), "rubric_score": (0, 2), "has_reading_passage": True},
            "essay_set_4": {"score1": (0, 2), "score2": (0, 2), "rubric_score": (0, 2), "has_reading_passage": True},
            "essay_set_5": {"score1": (0, 3), "score2": (0, 3), "rubric_score": (0, 3), "has_reading_passage": False},
            "essay_set_6": {"score1": (0, 3), "score2": (0, 3), "rubric_score": (0, 3), "has_reading_passage": False},
            "essay_set_7": {"score1": (0, 2), "score2": (0, 2), "rubric_score": (0, 2), "has_reading_passage": True},
            "essay_set_8": {"score1": (0, 2), "score2": (0, 2), "rubric_score": (0, 2), "has_reading_passage": False},
            "essay_set_9": {"score1": (0, 2), "score2": (0, 2), "rubric_score": (0, 2), "has_reading_passage": True},
            "essay_set_10": {"score1": (0, 2), "score2": (0, 2), "rubric_score": (0, 2), "has_reading_passage": False}
        }
    ),
    "rice_chem": DatasetConfig(
        name="rice_chem",
        description="Rice Chemistry dataset - 4 questions with rubric-based True/False scoring",
        huggingface_id="nlpatunt/rice_chem",
        evaluation_metrics=["accuracy", "Precision", "recall", "F1 score"],
        score_ranges={
            "question_1": {"score": (0, 8), "rubric_score": (0, 8)},  
            "question_2": {"score": (0, 9), "rubric_score": (0, 6)},     
            "question_3": {"score": (0, 9), "rubric_score": (0, 7)},     
            "question_4": {"score": (0, 8), "rubric_score": (0, 6)}       
        }
    ),

    "CSEE": DatasetConfig(
        name="CSEE",
        description="Computer Science Essay Evaluation dataset",
        huggingface_id="nlpatunt/CSEE",  # update with actual ID
        evaluation_metrics=["p-values", "quadratic_weighted_kappa"],
        score_ranges={
            "content_score": (0, 8),
            "language_score": (0, 8), 
            "structure_score": (0, 4),
            "overall_score": (0, 20)
        }
    ),
    "EFL": DatasetConfig(
        name="EFL",
        description="English as Foreign Language dataset - essays scored on 5 criteria (Grammar, Content, Organization, Style, Mechanics) with multi-rater evaluation",
        huggingface_id="nlpatunt/EFL",
        evaluation_metrics=["intraclass_correlation_coefficient"],
        score_ranges={
            "grammar": (1, 5),
            "content": (1, 5), 
            "organization": (1, 4),  # Only goes to 4 points based on rubric
            "style": (1, 5),
            "mechanics": (1, 5),
            "human_mean": (1.0, 5.0),  # Average of all rater scores
            "chatGPT": (1.0, 5.0),
            "brad": (1.0, 5.0)  # R1-R15 individual rater scores
        }
    ),
    "grade_like_a_human_dataset_os": DatasetConfig(
        name="tutorialCriteria",
        description="Tutorial-based grading dataset across six questions (Q1–Q6). Each subset includes multiple rater scores, and is used to assess grading consistency and reliability using regression metrics.",
        huggingface_id="nlpatunt/grade_like_a_human_dataset_os",
        evaluation_metrics=["mae", "rmse", "nrmse", "pearson", "accuracy"],
        score_ranges={
            "Q1": {"score": (0.0, 19.0)},
            "Q2": {"score": (0.0, 16.0)},
            "Q3": {"score": (0.0, 15.0)},
            "Q4": {"score": (0.0, 16.0)},
            "Q5": {"score": (0.0, 27.0)},
            "Q6": {"score": (0.0, 40.0)}
        }
    ),
    "persuade_2": DatasetConfig(
        name="persuade_2",
        description="Persuasive Essay dataset version 2 – essays written by students in grades 6–12, scored on a 1–6 scale using a rubric. Includes 15 different prompts.",
        huggingface_id="nlpatunt/persuade_2",  # Update if different
        evaluation_metrics=["quadratic_weighted_kappa", "mean_absolute_error_distance"],
        score_ranges={
            "Phones and driving": {"score": (1, 6), "rubric_score": (1, 6)},
            "Car-free cities": {"score": (1, 6), "rubric_score": (1, 6)},
            "Summer projects": {"score": (1, 6), "rubric_score": (1, 6)},
            "A Cowboy Who Rode the Waves": {"score": (1, 6), "rubric_score": (1, 6)},
            "Mandatory extracurricular activities": {"score": (1, 6), "rubric_score": (1, 6)},
            "Exploring Venus": {"score": (1, 6), "rubric_score": (1, 6)},
            "Facial action coding system": {"score": (1, 6), "rubric_score": (1, 6)},
            "The Face on Mars": {"score": (1, 6), "rubric_score": (1, 6)},
            "Community service": {"score": (1, 6), "rubric_score": (1, 6)},
            "Grades for extracurricular activities": {"score": (1, 6), "rubric_score": (1, 6)},
            "Driverless cars": {"score": (1, 6), "rubric_score": (1, 6)},
            "Does the electoral college work?": {"score": (1, 6), "rubric_score": (1, 6)},
            "Cell phones at school": {"score": (1, 6), "rubric_score": (1, 6)},
            "Distance learning": {"score": (1, 6), "rubric_score": (1, 6)},
            "Seeking multiple opinions": {"score": (1, 6), "rubric_score": (1, 6)}
        }
    ),

    
    # Datasets without rubrics
    "ASAP2": DatasetConfig(
        name="ASAP2",
        description="ASAP version 2 dataset - Evidence-based essay writing with source texts",
        huggingface_id="nlpatunt/ASAP2",  # Available on Kaggle
        evaluation_metrics=["quadratic_weighted_kappa","inter_rater_agreement"],
        score_ranges={
            "exploring_venus": {"holistic_score": (0, 4), "source_texts": 1},
            "cowboy_who_rode_waves": {"holistic_score": (0, 4), "source_texts": 1},
            "car_free_cities": {"holistic_score": (0, 4), "source_texts": 4},
            "electoral_college_work": {"holistic_score": (0, 4), "source_texts": 3},
            "driverless_cars": {"holistic_score": (0, 4), "source_texts": 1},
            "facial_action_coding": {"holistic_score": (0, 4), "source_texts": 1},
            "face_on_mars": {"holistic_score": (0, 4), "source_texts": 1}
        }
    ),
    "ASAP_plus_plus": DatasetConfig(
        name="ASAP_plus_plus",
        description="ASAP Plus Plus enhanced dataset – subset of ASAP-AES with 6 prompts and rubric scores",
        huggingface_id="nlpatunt/ASAP_plus_plus",
        evaluation_metrics=["QWK", "pearson_correlation"],
        score_ranges={
            "prompt_1": {"holistic_score": (2, 12), "rubric_score": (1, 6)},
            "prompt_2": {"holistic_score": (1, 6), "rubric_score": (1, 6)},
            "prompt_3": {"holistic_score": (0, 3), "rubric_score": (0, 3)},
            "prompt_4": {"holistic_score": (0, 3), "rubric_score": (0, 3)},
            "prompt_5": {"holistic_score": (0, 4), "rubric_score": (0, 4)},
            "prompt_6": {"holistic_score": (0, 4), "rubric_score": (0, 4)},
        }
    ),
    "SciEntSBank": DatasetConfig(
        name="SciEntSBank",
        description="Science Entailment Bank dataset - 135 questions in 2-way and 3-way classification formats",
        huggingface_id="nlpatunt/SciEntSBank",
        evaluation_metrics=["precision", "recall", "f1_score"],
        score_ranges={
            "2_way_classification": {
                "labels": ["correct", "incorrect"],
                "precision": (0.0, 1.0),
                "recall": (0.0, 1.0),
                "f1_score": (0.0, 1.0)
            },
            "3_way_classification": {
                "labels": ["correct", "incorrect_partial", "contradictory"],
                "precision": (0.0, 1.0),
                "recall": (0.0, 1.0),
                "f1_score": (0.0, 1.0)
            }
        }
    ),
    "BEEtlE": DatasetConfig(
        name="BEEtlE",
        description="BEEtlE dataset for educational assessment - classification-based evaluation",
        huggingface_id="nlpatunt/BEEtlE",  # update with actual ID if available
        evaluation_metrics=["precision", "recall", "f1_score"],
        score_ranges={
            "2_way_classification": {
                "labels": ["correct", "incorrect"],
                "precision": (0.0, 1.0),
                "recall": (0.0, 1.0),
                "f1_score": (0.0, 1.0)
            },
            "3_way_classification": {
                "labels": ["correct", "incorrect_partial", "contradictory"],
                "precision": (0.0, 1.0),
                "recall": (0.0, 1.0),
                "f1_score": (0.0, 1.0)
            }
        }
    ),
    "automatic_short_answer_grading_mohlar": DatasetConfig(
        name="automatic_short_answer_grading_mohlar",
        description="Automatic Short Answer Grading dataset - 2273 data points with expected answers, scored out of 5",
        huggingface_id="nlpatunt/automatic_short_answer_grading_mohlar",
        evaluation_metrics=["inter_annotator_agreement_percentage", "kappa_score", "weighted_kappa_score", "pearson_correlation"],
        score_ranges={
            "total_grade": (0, 5)
            
        }
    ),
}

def get_dataset_config(dataset_name: str) -> Optional[DatasetConfig]:
    return DATASETS.get(dataset_name)

def get_all_datasets() -> Dict[str, DatasetConfig]:
    return DATASETS