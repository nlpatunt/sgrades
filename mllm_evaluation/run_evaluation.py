#!/usr/bin/env python3
"""
Quick start script for BESESR-MLLM evaluation
"""

import asyncio
import os
import sys
from pathlib import Path
import requests

# Add current directory to path
sys.path.append(str(Path.cwd()))

def check_besesr_connection():
    """Check if BESESR is running"""
    try:
        response = requests.get("http://localhost:8000/health", timeout=5)
        return response.status_code == 200
    except:
        return False

def check_api_keys():
    """Check which API keys are available"""
    available_models = {}
    
    if os.getenv("OPENAI_API_KEY"):
        available_models["GPT-4o"] = {
            "model_id": "gpt-4o",
            "type": "api",
            "api_key_env": "OPENAI_API_KEY"
        }
        print("✓ OpenAI API key found - will test GPT-4o")
    
    if os.getenv("ANTHROPIC_API_KEY"):
        available_models["Claude-3.5-Sonnet"] = {
            "model_id": "claude-3.5-sonnet",
            "type": "api",
            "api_key_env": "ANTHROPIC_API_KEY"
        }
        print("✓ Anthropic API key found - will test Claude-3.5-Sonnet")
    
    if not available_models:
        print("No API keys found - will use simulation mode")
        available_models["TestModel"] = {
            "model_id": "simulation",
            "type": "simulation"
        }
    
    return available_models

async def quick_evaluation():
    """Run a quick evaluation"""
    
    print("Starting Quick BESESR-MLLM Evaluation")
    print("=" * 40)
    
    # Check BESESR connection
    if not check_besesr_connection():
        print("❌ BESESR platform not accessible at http://localhost:8000")
        print("Please start BESESR first with: uvicorn app.main:app --host 0.0.0.0 --port 8000")
        return
    
    print("✓ BESESR platform is accessible")
    
    # Check available models
    available_models = check_api_keys()
    
    # Try to import besesr_integration
    try:
        from besesr_integration import BESESRIntegration
        print("✓ BESESR integration loaded")
    except ImportError as e:
        print(f"❌ Failed to import besesr_integration: {e}")
        print("Make sure besesr_integration.py is in the current directory")
        return
    
    # Initialize integration
    integration = BESESRIntegration()
    
    # Test on 2-3 datasets only
    test_datasets = ["ASAP-AES", "BEEtlE_2way"]
    
    print(f"Testing {len(available_models)} models on {len(test_datasets)} datasets")
    
    # Run evaluation
    try:
        results = await integration.run_full_evaluation_pipeline(
            model_configs=available_models,
            dataset_names=test_datasets,
            max_essays_per_dataset=5,  # Small sample for quick test
            submit_to_platform=False    # Don't submit for quick test
        )
        
        print("\nQuick Evaluation Complete!")
        
        if results:
            leaderboard = integration.tester.generate_leaderboard(results)
            print("\nRankings:")
            for i, (model, score) in enumerate(leaderboard.items(), 1):
                print(f"{i}. {model}: {score:.3f}")
        
    except Exception as e:
        print(f"Evaluation failed: {e}")

def main():
    """Main entry point"""
    print("BESESR-MLLM Quick Start")
    print("=" * 30)
    
    # Run evaluation
    asyncio.run(quick_evaluation())

if __name__ == "__main__":
    main()