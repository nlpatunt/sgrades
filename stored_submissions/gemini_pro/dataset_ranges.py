#!/usr/bin/env python3
"""
Centralized dataset range configuration
Single source of truth for all evaluation scripts
"""

def get_score_range_for_dataset(dataset_name: str, essay_set: int = 1) -> str:
    """
    Get score range for dataset with proper ASAP-AES essay set handling
    
    Args:
        dataset_name: Name of dataset (e.g., "D_ASAP-AES")
        essay_set: Essay set number (1-8 for ASAP-AES)
    
    Returns:
        String representation of range (e.g., "2-12")
    """
    
    # ASAP-AES: 8 different essay sets with different ranges
    if dataset_name in ["ASAP-AES", "D_ASAP-AES"]:
        asap_ranges = {
            1: "2-12",   # Persuasive essays
            2: "1-6",    # Persuasive with source material
            3: "0-3",    # Source-dependent responses
            4: "0-3",    # Source-dependent responses
            5: "0-4",    # Source-dependent responses
            6: "0-4",    # Source-dependent responses
            7: "0-30",   # Narrative essays
            8: "0-60"    # Narrative essays
        }
        return asap_ranges.get(essay_set, "2-12")  # Default to Set 1
    
    # All other datasets - fixed ranges
    other_ranges = {
        "D_ASAP2": "0-4",
        "D_ASAP-SAS": "0-3",
        "D_ASAP_plus_plus": "0-60",
        "D_CSEE": "0-16",
        "D_BEEtlE_2way": "0-1",  # Binary: incorrect/correct
        "D_BEEtlE_3way": "0-2",  # Three-way: incorrect/contradictory/correct
        "D_SciEntSBank_2way": "0-1",
        "D_SciEntSBank_3way": "0-2",
        "D_Mohlar": "0-5",
        "D_Ielts_Writing_Dataset": "1-9",
        "D_Ielst_Writing_Task_2_Dataset": "1-9",
        "D_grade_like_a_human_dataset_os_q1": "0-19",
        "D_grade_like_a_human_dataset_os_q2": "0-16",
        "D_grade_like_a_human_dataset_os_q3": "0-15",
        "D_grade_like_a_human_dataset_os_q4": "0-16",
        "D_grade_like_a_human_dataset_os_q5": "0-27",
        "D_persuade_2": "1-6",
        "D_Regrading_Dataset_J2C": "0-8",
        "D_Rice_Chem_Q1": "0-8",
        "D_Rice_Chem_Q2": "0-8",
        "D_Rice_Chem_Q3": "0-9",
        "D_Rice_Chem_Q4": "0-8"
    }
    
    return other_ranges.get(dataset_name, "0-100")  # Generic fallback

def get_range_description(dataset_name: str, essay_set: int = 1) -> str:
    """Get human-readable description of the scoring range"""
    
    if dataset_name in ["ASAP-AES", "D_ASAP-AES"]:
        descriptions = {
            1: "2-12 scale (Persuasive essays)",
            2: "1-6 scale (Persuasive with sources)",
            3: "0-3 scale (Source-dependent)",
            4: "0-3 scale (Source-dependent)",
            5: "0-4 scale (Source-dependent)",
            6: "0-4 scale (Source-dependent)",
            7: "0-30 scale (Narrative essays)",
            8: "0-60 scale (Narrative essays)"
        }
        return descriptions.get(essay_set, "2-12 scale")
    
    range_str = get_score_range_for_dataset(dataset_name, essay_set)
    return f"{range_str} scale"
