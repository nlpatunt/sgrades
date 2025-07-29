"""
Global cache for storing evaluation results
"""

# Global dictionary to store results
EVALUATION_RESULTS = {}

def save_evaluation_result(model_id, result_data):
    """Save evaluation result to global cache"""
    EVALUATION_RESULTS[model_id] = result_data
    print(f"💾 Saved result for model {model_id} to global cache")
    print(f"📊 Cache now contains: {len(EVALUATION_RESULTS)} models")

def get_all_evaluation_results():
    """Get all evaluation results from cache"""
    return list(EVALUATION_RESULTS.values())

def get_evaluation_result(model_id):
    """Get specific evaluation result by model ID"""
    return EVALUATION_RESULTS.get(model_id)

def clear_cache():
    """Clear all cached results (for testing)"""
    global EVALUATION_RESULTS
    EVALUATION_RESULTS = {}
    print("🗑️ Cleared evaluation results cache")