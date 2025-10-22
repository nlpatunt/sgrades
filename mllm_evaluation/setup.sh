#!/bin/bash
# BESESR-MLLM Setup Script
# Setup environment for evaluating MLLM models on BESESR platform

set -e

echo "============================================="
echo "BESESR-MLLM Evaluation Setup"
echo "============================================="

# Check if Python 3.8+ is available
echo "Checking Python version..."
python_version=$(python3 --version 2>/dev/null | cut -d" " -f2 | cut -d"." -f1,2)
required_version="3.8"

if ! python3 -c "import sys; exit(0 if sys.version_info >= (3,8) else 1)" 2>/dev/null; then
    echo "Error: Python 3.8+ is required. Found: $python_version"
    exit 1
fi

echo "✓ Python version check passed"

# Create virtual environment
echo "Creating virtual environment..."
if [ ! -d "mllm_env" ]; then
    python3 -m venv mllm_env
    echo "✓ Virtual environment created"
else
    echo "✓ Virtual environment already exists"
fi

# Activate virtual environment
echo "Activating virtual environment..."
source mllm_env/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install PyTorch (CUDA version if available)
echo "Installing PyTorch..."
if command -v nvidia-smi &> /dev/null; then
    echo "CUDA detected, installing CUDA version of PyTorch..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
else
    echo "No CUDA detected, installing CPU version of PyTorch..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
fi

# Install transformers and related packages
echo "Installing Transformers and model dependencies..."
pip install transformers>=4.35.0
pip install accelerate>=0.21.0
pip install bitsandbytes>=0.41.0
pip install datasets>=2.14.0
pip install pillow>=9.5.0
pip install requests>=2.31.0

# Install API clients for closed-source models
echo "Installing API clients..."
pip install openai>=1.0.0
pip install anthropic>=0.3.0
pip install google-generativeai>=0.3.0

# Install evaluation and data processing libraries
echo "Installing evaluation libraries..."
pip install pandas>=1.5.0
pip install numpy>=1.24.0
pip install scikit-learn>=1.3.0
pip install scipy>=1.11.0
pip install matplotlib>=3.7.0
pip install seaborn>=0.12.0

# Install additional utilities
echo "Installing additional utilities..."
pip install tqdm>=4.65.0
pip install python-dotenv>=1.0.0
pip install aiohttp>=3.8.0
pip install asyncio-throttle>=1.0.0

# Create directory structure
echo "Creating directory structure..."
mkdir -p models/
mkdir -p data/
mkdir -p results/
mkdir -p configs/
mkdir -p logs/

# Create environment file template
echo "Creating environment file template..."
cat > .env << EOF
# API Keys for Closed-Source Models
OPENAI_API_KEY=your_openai_api_key_here
ANTHROPIC_API_KEY=your_anthropic_api_key_here
GOOGLE_API_KEY=your_google_api_key_here

# BESESR Platform Settings
BESESR_BASE_URL=http://localhost:8000
MAX_ESSAYS_PER_DATASET=50
EVALUATION_BATCH_SIZE=10

# Model Settings
MODEL_CACHE_DIR=./models/
RESULTS_DIR=./results/
LOG_LEVEL=INFO

# Hardware Settings (adjust based on your setup)
CUDA_VISIBLE_DEVICES=0
HF_HOME=./models/huggingface/
TRANSFORMERS_CACHE=./models/transformers_cache/
EOF

# Create configuration file for models
echo "Creating model configuration file..."
cat > configs/model_config.yaml << EOF
# Model Configuration for BESESR Evaluation
# Based on EssayJudge paper models

open_source_models:
  yi_vl_6b:
    model_id: "01-ai/Yi-VL-6B"
    model_type: "vision_language"
    memory_requirement: "16GB"
    notes: "Yi-VL 6B parameter model"
    
  qwen2_vl_7b:
    model_id: "Qwen/Qwen2-VL-7B"
    model_type: "vision_language"
    memory_requirement: "18GB"
    notes: "Qwen2-VL 7B parameter model"
    
  deepseek_vl_7b:
    model_id: "deepseek-ai/deepseek-vl-7b-chat"
    model_type: "vision_language"
    memory_requirement: "18GB"
    notes: "DeepSeek-VL 7B parameter model"
    
  llava_next_8b:
    model_id: "llava-hf/llava-v1.6-mistral-7b-hf"
    model_type: "vision_language"
    memory_requirement: "20GB"
    notes: "LLaVA-NEXT 8B parameter model"
    
  internvl2_8b:
    model_id: "OpenGVLab/InternVL2-8B"
    model_type: "vision_language"
    memory_requirement: "20GB"
    notes: "InternVL2 8B parameter model"

closed_source_models:
  gpt4o:
    model_id: "gpt-4o"
    api_provider: "openai"
    cost_per_1k_tokens: 0.005
    notes: "GPT-4o with vision capabilities"
    
  claude_35_sonnet:
    model_id: "claude-3-5-sonnet-20241022"
    api_provider: "anthropic"
    cost_per_1k_tokens: 0.003
    notes: "Claude 3.5 Sonnet with vision capabilities"
    
  gemini_15_pro:
    model_id: "gemini-1.5-pro"
    api_provider: "google"
    cost_per_1k_tokens: 0.00125
    notes: "Gemini 1.5 Pro with vision capabilities"

evaluation_settings:
  traits:
    - "lexical_accuracy"
    - "lexical_diversity"
    - "grammatical_accuracy"
    - "grammatical_diversity"
    - "punctuation_accuracy"
    - "coherence"
    - "argument_clarity"
    - "justifying_persuasiveness"
    - "organizational_structure"
    - "essay_length"
    
  datasets:
    - "ASAP-AES"
    - "BEEtlE_2way"
    - "BEEtlE_3way"
    - "CSEE"
    - "EFL"
    - "SciEntSBank_2way"
    - "SciEntSBank_3way"
    - "persuade_2"
    - "grade_like_a_human_dataset_os_q1"
    - "Rice_Chem_Q1"
EOF

# Create requirements.txt for easy setup
echo "Creating requirements.txt..."
cat > requirements.txt << EOF
# Core ML libraries
torch>=2.0.0
torchvision>=0.15.0
torchaudio>=2.0.0
transformers>=4.35.0
accelerate>=0.21.0
bitsandbytes>=0.41.0

# Data processing
pandas>=1.5.0
numpy>=1.24.0
datasets>=2.14.0
pillow>=9.5.0

# API clients
openai>=1.0.0
anthropic>=0.3.0
google-generativeai>=0.3.0

# HTTP and async
requests>=2.31.0
aiohttp>=3.8.0
asyncio-throttle>=1.0.0

# Evaluation and visualization
scikit-learn>=1.3.0
scipy>=1.11.0
matplotlib>=3.7.0
seaborn>=0.12.0

# Utilities
tqdm>=4.65.0
python-dotenv>=1.0.0
pyyaml>=6.0.0
EOF

# Create a simple test script
echo "Creating test script..."
cat > test_setup.py << EOF
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
        'torch',
        'transformers', 
        'pandas',
        'numpy',
        'requests',
        'PIL',
        'sklearn',
        'matplotlib'
    ]
    
    # Optional API libraries
    optional_modules = [
        'openai',
        'anthropic', 
        'google.generativeai'
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
        
        # Test CUDA availability
        import torch
        if torch.cuda.is_available():
            print(f"✓ CUDA available: {torch.cuda.device_count()} GPU(s)")
            print(f"  GPU: {torch.cuda.get_device_name(0)}")
        else:
            print("ℹ CUDA not available (will use CPU)")
        
        return True
    else:
        print("✗ Setup verification failed!")
        print("Please run: pip install -r requirements.txt")
        return False

if __name__ == "__main__":
    main()
EOF

# Create quick start script
echo "Creating quick start script..."
cat > run_evaluation.py << EOF
#!/usr/bin/env python3
"""
Quick start script for BESESR-MLLM evaluation
"""

import asyncio
import os
import sys
from pathlib import Path

# Add current directory to path
sys.path.append(str(Path.cwd()))

from besesr_integration import BESESRIntegration

async def quick_evaluation():
    """Run a quick evaluation with a subset of models and datasets"""
    
    print("Starting Quick BESESR-MLLM Evaluation")
    print("=" * 40)
    
    # Initialize integration
    integration = BESESRIntegration()
    
    # Quick configuration - only test available models
    quick_configs = {}
    
    # Check for API keys
    if os.getenv("OPENAI_API_KEY"):
        quick_configs["GPT-4o"] = {
            "model_id": "gpt-4o",
            "type": "api",
            "api_key_env": "OPENAI_API_KEY"
        }
        print("✓ Will test GPT-4o")
    
    if os.getenv("ANTHROPIC_API_KEY"):
        quick_configs["Claude-3.5-Sonnet"] = {
            "model_id": "claude-3.5-sonnet",
            "type": "api",
            "api_key_env": "ANTHROPIC_API_KEY"
        }
        print("✓ Will test Claude-3.5-Sonnet")
    
    # Try one local model if no API keys available
    if not quick_configs:
        print("No API keys found, trying local model...")
        quick_configs["Qwen2-VL-7B"] = {
            "model_id": "qwen2-vl",
            "type": "local"
        }
    
    if not quick_configs:
        print("No models configured. Please set API keys or install local models.")
        return
    
    # Test on 2-3 datasets only
    test_datasets = ["ASAP-AES", "BEEtlE_2way", "CSEE"]
    
    print(f"Testing {len(quick_configs)} models on {len(test_datasets)} datasets")
    
    # Run evaluation
    try:
        results = await integration.run_full_evaluation_pipeline(
            model_configs=quick_configs,
            dataset_names=test_datasets,
            max_essays_per_dataset=10,  # Small sample for quick test
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
        print("Check that BESESR platform is running at http://localhost:8000")

def main():
    """Main entry point"""
    print("BESESR-MLLM Quick Start")
    print("=" * 30)
    
    # Check if .env file exists
    if not os.path.exists('.env'):
        print("Creating .env file from template...")
        print("Please edit .env file with your API keys before running evaluation.")
        return
    
    # Load environment variables
    from dotenv import load_dotenv
    load_dotenv()
    
    # Run evaluation
    asyncio.run(quick_evaluation())

if __name__ == "__main__":
    main()
EOF

# Create documentation
echo "Creating README..."
cat > README.md << EOF
# BESESR-MLLM Evaluation Framework

This framework evaluates Multimodal Large Language Models (MLLMs) on the BESESR automated essay scoring platform, based on the methodology from the EssayJudge paper.

## Quick Start

1. **Setup Environment:**
   \`\`\`bash
   ./setup.sh
   source mllm_env/bin/activate
   \`\`\`

2. **Configure API Keys (optional):**
   Edit \`.env\` file with your API keys for closed-source models:
   \`\`\`
   OPENAI_API_KEY=your_key_here
   ANTHROPIC_API_KEY=your_key_here  
   GOOGLE_API_KEY=your_key_here
   \`\`\`

3. **Test Setup:**
   \`\`\`bash
   python test_setup.py
   \`\`\`

4. **Start BESESR Platform:**
   Make sure your BESESR platform is running at http://localhost:8000

5. **Run Quick Evaluation:**
   \`\`\`bash
   python run_evaluation.py
   \`\`\`

## Supported Models

### Open-Source Models
- Yi-VL 6B
- Qwen2-VL 7B  
- DeepSeek-VL 7B
- LLaVA-NEXT 8B
- InternVL2 8B
- MiniCPM-V2.6 8B
- LLaMA-3.2-Vision 11B

### Closed-Source Models (API Required)
- GPT-4o
- Claude-3.5-Sonnet
- Gemini-1.5-Pro

## Evaluation Traits

Based on EssayJudge paper methodology:

### Lexical Level
- Lexical Accuracy
- Lexical Diversity

### Sentence Level  
- Grammatical Accuracy
- Grammatical Diversity
- Punctuation Accuracy
- Coherence

### Discourse Level
- Argument Clarity
- Justifying Persuasiveness
- Organizational Structure
- Essay Length

## BESESR Datasets

Evaluates on all 25+ datasets including:
- ASAP-AES, ASAP2, ASAP++
- BEEtlE (2-way, 3-way)
- SciEntSBank (2-way, 3-way)
- CSEE, EFL, Mohlar
- Grade Like Human (OS Q1-Q5)
- Rice Chemistry (Q1-Q4)
- And more...

## Hardware Requirements

### For Local Models:
- **Minimum:** 16GB RAM, 8GB VRAM
- **Recommended:** 32GB RAM, 16GB+ VRAM
- **Optimal:** 64GB RAM, 24GB+ VRAM

### For API Models Only:
- **Minimum:** 8GB RAM
- Internet connection for API calls

## Usage Examples

### Custom Evaluation:
\`\`\`python
from besesr_integration import BESESRIntegration

# Initialize
integration = BESESRIntegration()

# Configure models
models = {
    "GPT-4o": {
        "model_id": "gpt-4o",
        "type": "api", 
        "api_key_env": "OPENAI_API_KEY"
    }
}

# Run evaluation
results = await integration.run_full_evaluation_pipeline(
    model_configs=models,
    dataset_names=["ASAP-AES", "CSEE"],
    max_essays_per_dataset=100,
    submit_to_platform=True
)
\`\`\`

### Generate Predictions Only:
\`\`\`python
# Load model
model = MLLMModelFactory.create_model("gpt-4o", api_key="your_key")
model.load_model()

# Generate predictions
csv_path = integration.generate_csv_predictions(
    "ASAP-AES", essays, model
)

# Submit to BESESR
integration.submit_predictions_to_besesr(
    csv_path, "ASAP-AES", "GPT-4o"
)
\`\`\`

## Output Files

- **Results:** \`./results/evaluation_results_TIMESTAMP.json\`
- **Reports:** \`./results/evaluation_report_TIMESTAMP.md\`
- **Predictions:** \`./results/MODEL_DATASET_predictions.csv\`
- **Logs:** \`./besesr_evaluation.log\`

## Troubleshooting

### Common Issues:

1. **CUDA Out of Memory:**
   - Reduce \`max_essays_per_dataset\`
   - Use smaller models
   - Enable gradient checkpointing

2. **API Rate Limits:**
   - Add delays between requests
   - Use cheaper models for testing
   - Check API quotas

3. **Model Loading Fails:**
   - Check available disk space
   - Verify model IDs
   - Check HuggingFace authentication

4. **BESESR Connection Issues:**
   - Verify platform is running
   - Check \`BESESR_BASE_URL\` in .env
   - Test API endpoints manually

## Contributing

1. Add new models to \`mllm_models.py\`
2. Update configurations in \`configs/model_config.yaml\`
3. Test with \`python test_setup.py\`
4. Submit evaluation results

## Citation

If you use this framework, please cite:

\`\`\`bibtex
@article{essayjudge2024,
  title={EssayJudge: A Multi-Granular Benchmark for Assessing Automated Essay Scoring Capabilities of Multimodal Large Language Models},
  author={Su, Jiamin and Yan, Yibo and Fu, Fangteng and Zhang, Han and Ye, Jingheng and Liu, Xiang and Huo, Jiahao and Zhou, Huiyu and Hu, Xuming},
  journal={arXiv preprint arXiv:2502.11916},
  year={2024}
}
\`\`\`

## License

This framework is provided for research purposes. Please respect the licenses of individual models and datasets.
EOF

# Make scripts executable
chmod +x test_setup.py
chmod +x run_evaluation.py

echo "✓ Setup complete!"
echo ""
echo "Next steps:"
echo "1. Activate virtual environment: source mllm_env/bin/activate"
echo "2. Test setup: python test_setup.py" 
echo "3. Edit .env file with your API keys (optional)"
echo "4. Make sure BESESR platform is running"
echo "5. Run evaluation: python run_evaluation.py"
echo ""
echo "For full documentation, see README.md"