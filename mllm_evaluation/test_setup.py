#!/usr/bin/env python3
"""
Test script to verify BESESR-MLLM setup
"""

import sys
import importlib

def test_import(module_name):
    try:
        importlib.import_module(module_name)
        print(f"✓ {module_name}")
        return True
    except ImportError as e:
        print(f"✗ {module_name}: {e}")
        return False

def main():
    print("Testing BESESR-MLLM Setup")
    print("=" * 30)
    
    # Core libraries
    modules_to_test = [
        'requests',
        'pandas',
        'numpy',
        'json'
    ]
    
    # Optional API libraries
    optional_modules = [
        'openai',
        'anthropic'
    ]
    
    success_count = 0
    
    print("Core modules:")
    for module in modules_to_test:
        if test_import(module):
            success_count += 1
            
    print(f"\nOptional modules (for API access):")
    for module in optional_modules:
        test_import(module)
    
    print(f"\nCore setup: {success_count}/{len(modules_to_test)} modules OK")
    
    if success_count == len(modules_to_test):
        print("✓ Setup verification passed!")
        
        # Test API keys
        import os
        api_keys = {
            'OPENAI_API_KEY': os.getenv('OPENAI_API_KEY'),
            'ANTHROPIC_API_KEY': os.getenv('ANTHROPIC_API_KEY')
        }
        
        print("\nAPI Keys:")
        for key_name, key_value in api_keys.items():
            if key_value and len(key_value) > 10:
                print(f"✓ {key_name}: {key_value[:10]}...")
            else:
                print(f"✗ {key_name}: Not set or invalid")
        
        return True
    else:
        print("✗ Setup verification failed!")
        return False

if __name__ == "__main__":
    main()