#!/usr/bin/env python3
"""
Centralized dataset range configuration
Single source of truth for all evaluation scripts
"""

def get_score_range_for_dataset(dataset_name: str, essay_set: int = 1) -> tuple:
    """
    Get score range for dataset with proper ASAP-AES essay set handling
    
    Args:
        dataset_name: Name of dataset (e.g., "CSEE" or "D_CSEE")
        essay_set: Essay set number (1-8 for ASAP-AES)
    
    Returns:
        Tuple of (min, max) score range
    """
    
    # ASAP-AES: 8 different essay sets with different ranges
    if dataset_name in ["ASAP-AES", "D_ASAP-AES"]:
        asap_ranges = {
            1: (2, 12),   # Persuasive essays
            2: (1, 6),    # Persuasive with source material
            3: (0, 3),    # Source-dependent responses
            4: (0, 3),    # Source-dependent responses
            5: (0, 4),    # Source-dependent responses
            6: (0, 4),    # Source-dependent responses
            7: (0, 30),   # Narrative essays
            8: (0, 60)    # Narrative essays
        }
        return asap_ranges.get(essay_set, (2, 12))  # Default to Set 1
    
    # ASAP2: Single range 0-3 for all essay sets
    elif dataset_name in ["ASAP2", "D_ASAP2"]:
        return (0, 3)
    
    # ASAP_plus_plus: Same as ASAP-AES (8 essay sets)
    elif dataset_name in ["ASAP_plus_plus", "D_ASAP_plus_plus"]:
        asap_plus_ranges = {
            1: (2, 12),   # Persuasive essays
            2: (1, 6),    # Persuasive with source material
            3: (0, 3),    # Source-dependent responses
            4: (0, 3),    # Source-dependent responses
            5: (0, 4),    # Source-dependent responses
            6: (0, 4),    # Source-dependent responses
            7: (0, 30),   # Narrative essays
            8: (0, 60)    # Narrative essays
        }
        return asap_plus_ranges.get(essay_set, (2, 12))
    
    # All other datasets - fixed ranges
    # Support both with and without "D_" prefix
    other_ranges = {
        # With D_ prefix
        "D_ASAP-SAS": (0, 3),
        "D_CSEE": (0, 16),
        "D_BEEtlE_2way": (0, 1),
        "D_BEEtlE_3way": (0, 2),
        "D_SciEntSBank_2way": (0, 1),
        "D_SciEntSBank_3way": (0, 2),
        "D_Mohlar": (0, 5),
        "D_Ielts_Writing_Dataset": (1, 9),
        "D_Ielts_Writing_Task_2_Dataset": (1, 9),
        "D_OS_Dataset_q1": (0, 19),
        "D_OS_Dataset_q2": (0, 16),
        "D_OS_Dataset_q3": (0, 15),
        "D_OS_Dataset_q4": (0, 16),
        "D_OS_Dataset_q5": (0, 27),
        "D_persuade_2": (1, 6),
        "D_Regrading_Dataset_J2C": (0, 8),
        "D_Rice_Chem_Q1": (0, 8),
        "D_Rice_Chem_Q2": (0, 8),
        "D_Rice_Chem_Q3": (0, 9),
        "D_Rice_Chem_Q4": (0, 8),
        
        # Without D_ prefix (duplicates for convenience)
        "ASAP-SAS": (0, 3),
        "CSEE": (0, 16),
        "BEEtlE_2way": (0, 1),
        "BEEtlE_3way": (0, 2),
        "SciEntSBank_2way": (0, 1),
        "SciEntSBank_3way": (0, 2),
        "Mohlar": (0, 5),
        "Ielts_Writing_Dataset": (1, 9),
        "Ielts_Writing_Task_2_Dataset": (1, 9),
        "OS_Dataset_q1": (0, 19),
        "OS_Dataset_q2": (0, 16),
        "OS_Dataset_q3": (0, 15),
        "OS_Dataset_q4": (0, 16),
        "OS_Dataset_q5": (0, 27),
        "persuade_2": (1, 6),
        "Regrading_Dataset_J2C": (0, 8),
        "Rice_Chem_Q1": (0, 8),
        "Rice_Chem_Q2": (0, 8),
        "Rice_Chem_Q3": (0, 9),
        "Rice_Chem_Q4": (0, 8)
    }
    
    return other_ranges.get(dataset_name, (0, 100))  # Generic fallback


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
    
    elif dataset_name in ["ASAP2", "D_ASAP2"]:
        return "0-3 scale (all essay sets)"
    
    elif dataset_name in ["ASAP_plus_plus", "D_ASAP_plus_plus"]:
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
    
    range_tuple = get_score_range_for_dataset(dataset_name, essay_set)
    return f"{range_tuple[0]}-{range_tuple[1]} scale"