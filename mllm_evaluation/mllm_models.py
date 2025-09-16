import torch
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, 
    LlamaTokenizer, LlamaForCausalLM,
    AutoProcessor, AutoModel
)
from PIL import Image
import requests
import base64
import json
import time
from typing import List, Dict, Any, Optional, Union
import logging

logger = logging.getLogger(__name__)

class BaseMLLMModel:
    """Base class for all MLLM models"""
    
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self.processor = None
        
    def load_model(self):
        """Load the model - to be implemented by subclasses"""
        raise NotImplementedError
        
    def generate_response(self, prompt: str, images: Optional[List[Image.Image]] = None) -> str:
        """Generate response - to be implemented by subclasses"""
        raise NotImplementedError
        
    def score_essay(self, essay_text: str, prompt: str, trait: str, images: Optional[List[Image.Image]] = None) -> float:
        """Score essay for a specific trait"""
        scoring_prompt = self._create_scoring_prompt(essay_text, prompt, trait)
        response = self.generate_response(scoring_prompt, images)
        return self._extract_score(response)
    
    def _create_scoring_prompt(self, essay_text: str, prompt: str, trait: str) -> str:
        """Create scoring prompt based on EssayJudge methodology"""
        trait_rubrics = {
            "lexical_accuracy": "Rate lexical accuracy (0-5): word choice, spelling, semantic appropriateness",
            "lexical_diversity": "Rate lexical diversity (0-5): vocabulary variety and richness",
            "grammatical_accuracy": "Rate grammatical accuracy (0-5): sentence structure correctness",
            "grammatical_diversity": "Rate grammatical diversity (0-5): variety of sentence structures",
            "punctuation_accuracy": "Rate punctuation accuracy (0-5): correct punctuation usage",
            "coherence": "Rate coherence (0-5): logical connections between sentences",
            "argument_clarity": "Rate argument clarity (0-5): clarity and focus of central argument",
            "justifying_persuasiveness": "Rate justifying persuasiveness (0-5): strength of evidence and reasoning",
            "organizational_structure": "Rate organizational structure (0-5): essay organization and paragraph structure",
            "essay_length": "Rate essay length (0-5): appropriate length and depth"
        }
        
        rubric = trait_rubrics.get(trait, f"Rate {trait} (0-5)")
        
        return f"""You are a professional English educator. Score this essay's {trait}.

TASK: {rubric}

ESSAY PROMPT: {prompt}

STUDENT ESSAY:
{essay_text}

Provide only a numerical score from 0-5 where:
5 = Excellent performance
4 = Good performance  
3 = Average performance
2 = Below average performance
1 = Poor performance
0 = Very poor performance

Score:"""

    def _extract_score(self, response: str) -> float:
        import re
        numbers = re.findall(r'\b[0-5](?:\.[0-9]+)?\b', response)
        if numbers:
            try:
                score = float(numbers[0])
                return min(5.0, max(0.0, score))  # Clamp to 0-5 range
            except ValueError:
                pass
        
        # Default fallback
        logger.warning(f"Could not extract score from response: {response[:100]}")
        return 2.5  # Average score as fallback

class YiVLModel(BaseMLLMModel):
    def __init__(self):
        super().__init__("Yi-VL-6B")
        
    def load_model(self):
        model_path = "01-ai/Yi-VL-6B"
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True
            )
            logger.info(f"Loaded {self.model_name}")
        except Exception as e:
            logger.error(f"Failed to load {self.model_name}: {e}")
            
    def generate_response(self, prompt: str, images: Optional[List[Image.Image]] = None) -> str:
        if not self.model:
            return "Model not loaded"
            
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt")
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs.input_ids,
                    max_length=200,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            return response[len(prompt):].strip()
        except Exception as e:
            logger.error(f"Generation failed for {self.model_name}: {e}")
            return "Error in generation"

class Qwen2VLModel(BaseMLLMModel):
    def __init__(self):
        super().__init__("Qwen2-VL-7B")
        
    def load_model(self):
        model_path = "Qwen/Qwen2-VL-7B"
        try:
            self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True
            )
            logger.info(f"Loaded {self.model_name}")
        except Exception as e:
            logger.error(f"Failed to load {self.model_name}: {e}")
            
    def generate_response(self, prompt: str, images: Optional[List[Image.Image]] = None) -> str:
        if not self.model or not self.processor:
            return "Model not loaded"
            
        try:
            if images:
                inputs = self.processor(text=prompt, images=images[0], return_tensors="pt")
            else:
                inputs = self.processor(text=prompt, return_tensors="pt")
                
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=200,
                    temperature=0.7,
                    do_sample=True
                )
            response = self.processor.decode(outputs[0], skip_special_tokens=True)
            return response[len(prompt):].strip()
        except Exception as e:
            logger.error(f"Generation failed for {self.model_name}: {e}")
            return "Error in generation"

class DeepSeekVLModel(BaseMLLMModel):
    def __init__(self):
        super().__init__("DeepSeek-VL-7B")
        
    def load_model(self):
        model_path = "deepseek-ai/deepseek-vl-7b-chat"
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True
            )
            logger.info(f"Loaded {self.model_name}")
        except Exception as e:
            logger.error(f"Failed to load {self.model_name}: {e}")

class LLaVANextModel(BaseMLLMModel):
    def __init__(self):
        super().__init__("LLaVA-NEXT-8B")
        
    def load_model(self):
        model_path = "llava-hf/llava-v1.6-mistral-7b-hf"
        try:
            self.processor = AutoProcessor.from_pretrained(model_path)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
                device_map="auto"
            )
            logger.info(f"Loaded {self.model_name}")
        except Exception as e:
            logger.error(f"Failed to load {self.model_name}: {e}")

class InternVL2Model(BaseMLLMModel):
    def __init__(self):
        super().__init__("InternVL2-8B")
        
    def load_model(self):
        model_path = "OpenGVLab/InternVL2-8B"
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            self.model = AutoModel.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True
            )
            logger.info(f"Loaded {self.model_name}")
        except Exception as e:
            logger.error(f"Failed to load {self.model_name}: {e}")

class MiniCPMV26Model(BaseMLLMModel):
    """MiniCPM-V2.6 8B implementation"""
    
    def __init__(self):
        super().__init__("MiniCPM-V2.6-8B")
        
    def load_model(self):
        model_path = "openbmb/MiniCPM-V-2_6"
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            self.model = AutoModel.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True
            )
            logger.info(f"Loaded {self.model_name}")
        except Exception as e:
            logger.error(f"Failed to load {self.model_name}: {e}")

class LLaMA32VisionModel(BaseMLLMModel):
    """LLaMA-3.2-Vision 11B implementation"""
    
    def __init__(self):
        super().__init__("LLaMA-3.2-Vision-11B")
        
    def load_model(self):
        model_path = "meta-llama/Llama-3.2-11B-Vision-Instruct"
        try:
            self.processor = AutoProcessor.from_pretrained(model_path)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
                device_map="auto"
            )
            logger.info(f"Loaded {self.model_name}")
        except Exception as e:
            logger.error(f"Failed to load {self.model_name}: {e}")

# =============================================================================
# CLOSED-SOURCE MODELS (API-based)
# =============================================================================

class GPT4oModel(BaseMLLMModel):
    """GPT-4o implementation via OpenAI API"""
    
    def __init__(self, api_key: str):
        super().__init__("GPT-4o")
        self.api_key = api_key
        self.client = None
        
    def load_model(self):
        try:
            import openai
            self.client = openai.OpenAI(api_key=self.api_key)
            logger.info(f"Loaded {self.model_name}")
        except Exception as e:
            logger.error(f"Failed to load {self.model_name}: {e}")
            
    def generate_response(self, prompt: str, images: Optional[List[Image.Image]] = None) -> str:
        if not self.client:
            return "Model not loaded"
            
        try:
            messages = [{"role": "user", "content": prompt}]
            
            # Add images if provided
            if images:
                content = [{"type": "text", "text": prompt}]
                for img in images[:1]:  # GPT-4o supports images
                    # Convert PIL Image to base64
                    import io
                    buffer = io.BytesIO()
                    img.save(buffer, format="PNG")
                    img_b64 = base64.b64encode(buffer.getvalue()).decode()
                    content.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{img_b64}"}
                    })
                messages = [{"role": "user", "content": content}]
            
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                max_tokens=200,
                temperature=0.7
            )
            
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Generation failed for {self.model_name}: {e}")
            return "Error in generation"

class Claude35SonnetModel(BaseMLLMModel):
    """Claude-3.5-Sonnet implementation via Anthropic API"""
    
    def __init__(self, api_key: str):
        super().__init__("Claude-3.5-Sonnet")
        self.api_key = api_key
        self.client = None
        
    def load_model(self):
        try:
            import anthropic
            self.client = anthropic.Anthropic(api_key=self.api_key)
            logger.info(f"Loaded {self.model_name}")
        except Exception as e:
            logger.error(f"Failed to load {self.model_name}: {e}")
            
    def generate_response(self, prompt: str, images: Optional[List[Image.Image]] = None) -> str:
        if not self.client:
            return "Model not loaded"
            
        try:
            content = [{"type": "text", "text": prompt}]
            
            # Add images if provided
            if images:
                for img in images[:1]:  # Claude supports images
                    import io
                    buffer = io.BytesIO()
                    img.save(buffer, format="PNG")
                    img_b64 = base64.b64encode(buffer.getvalue()).decode()
                    content.append({
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": img_b64
                        }
                    })
            
            response = self.client.messages.create(
                model="claude-3-5-sonnet-20241220" ,
                max_tokens=200,
                messages=[{"role": "user", "content": content}]
            )
            
            return response.content[0].text
        except Exception as e:
            logger.error(f"Generation failed for {self.model_name}: {e}")
            return "Error in generation"

class GeminiModel(BaseMLLMModel):
    """Gemini 1.5 Pro implementation via Google AI API"""
    
    def __init__(self, api_key: str):
        super().__init__("Gemini-1.5-Pro")
        self.api_key = api_key
        self.model = None
        
    def load_model(self):
        try:
            import google.generativeai as genai
            genai.configure(api_key=self.api_key)
            self.model = genai.GenerativeModel('gemini-1.5-pro')
            logger.info(f"Loaded {self.model_name}")
        except Exception as e:
            logger.error(f"Failed to load {self.model_name}: {e}")
            
    def generate_response(self, prompt: str, images: Optional[List[Image.Image]] = None) -> str:
        if not self.model:
            return "Model not loaded"
            
        try:
            if images:
                response = self.model.generate_content([prompt, images[0]])
            else:
                response = self.model.generate_content(prompt)
            
            return response.text
        except Exception as e:
            logger.error(f"Generation failed for {self.model_name}: {e}")
            return "Error in generation"

# =============================================================================
# MODEL FACTORY AND TESTING FRAMEWORK
# =============================================================================

class MLLMModelFactory:
    """Factory for creating MLLM models"""
    
    @staticmethod
    def create_model(model_name: str, **kwargs) -> BaseMLLMModel:
        """Create a model instance by name"""
        model_map = {
            "yi-vl": YiVLModel,
            "qwen2-vl": Qwen2VLModel,
            "deepseek-vl": DeepSeekVLModel,
            "llava-next": LLaVANextModel,
            "internvl2": InternVL2Model,
            "minicpm-v26": MiniCPMV26Model,
            "llama32-vision": LLaMA32VisionModel,
            "gpt-4o": lambda: GPT4oModel(kwargs.get("api_key")),
            "claude-3.5-sonnet": lambda: Claude35SonnetModel(kwargs.get("api_key")),
            "gemini-1.5-pro": lambda: GeminiModel(kwargs.get("api_key"))
        }
        
        if model_name.lower() not in model_map:
            raise ValueError(f"Unknown model: {model_name}")
            
        return model_map[model_name.lower()]()

class BESESRModelTester:
    """Test framework for evaluating models on BESESR"""
    
    def __init__(self):
        self.models: Dict[str, BaseMLLMModel] = {}
        self.results: Dict[str, Dict] = {}
        
    def add_model(self, model: BaseMLLMModel):
        """Add a model to test"""
        self.models[model.model_name] = model
        
    def evaluate_model_on_dataset(self, 
                                model_name: str, 
                                dataset_name: str, 
                                essays: List[Dict],
                                traits: List[str] = None) -> Dict[str, float]:
        """Evaluate a model on a specific dataset"""
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")
            
        model = self.models[model_name]
        
        # Default traits from EssayJudge paper
        if traits is None:
            traits = [
                "lexical_accuracy", "lexical_diversity", "grammatical_accuracy",
                "grammatical_diversity", "punctuation_accuracy", "coherence",
                "argument_clarity", "justifying_persuasiveness", 
                "organizational_structure", "essay_length"
            ]
        
        results = {trait: [] for trait in traits}
        
        logger.info(f"Evaluating {model_name} on {dataset_name} with {len(essays)} essays")
        
        for i, essay_data in enumerate(essays):
            essay_text = essay_data.get("essay_text", "")
            prompt = essay_data.get("prompt", "")
            images = essay_data.get("images", [])
            
            for trait in traits:
                try:
                    score = model.score_essay(essay_text, prompt, trait, images)
                    results[trait].append(score)
                    
                    if i < 5:  # Log first few for debugging
                        logger.info(f"{model_name} - Essay {i+1} - {trait}: {score}")
                        
                except Exception as e:
                    logger.error(f"Error scoring essay {i+1} for {trait}: {e}")
                    results[trait].append(2.5)  # Default score
            
            # Progress logging
            if (i + 1) % 10 == 0:
                logger.info(f"Processed {i+1}/{len(essays)} essays")
        
        # Calculate average scores
        avg_results = {trait: sum(scores)/len(scores) if scores else 0 
                      for trait, scores in results.items()}
        
        self.results[f"{model_name}_{dataset_name}"] = avg_results
        return avg_results
    
    def run_comprehensive_evaluation(self, datasets: Dict[str, List[Dict]]) -> Dict[str, Dict]:
        """Run evaluation across all models and datasets"""
        all_results = {}
        
        for model_name in self.models:
            model_results = {}
            for dataset_name, essays in datasets.items():
                try:
                    results = self.evaluate_model_on_dataset(model_name, dataset_name, essays)
                    model_results[dataset_name] = results
                except Exception as e:
                    logger.error(f"Failed to evaluate {model_name} on {dataset_name}: {e}")
                    
            all_results[model_name] = model_results
            
        return all_results
    
    def generate_leaderboard(self, results: Dict[str, Dict]) -> Dict[str, float]:
        """Generate overall leaderboard based on average performance"""
        leaderboard = {}
        
        for model_name, model_results in results.items():
            all_scores = []
            for dataset_results in model_results.values():
                all_scores.extend(dataset_results.values())
            
            if all_scores:
                leaderboard[model_name] = sum(all_scores) / len(all_scores)
            else:
                leaderboard[model_name] = 0.0
                
        return dict(sorted(leaderboard.items(), key=lambda x: x[1], reverse=True))

# =============================================================================
# USAGE EXAMPLE
# =============================================================================

def main():
    """Example usage of the MLLM testing framework"""
    
    # Initialize tester
    tester = BESESRModelTester()
    
    # Load models (you'll need API keys for closed-source models)
    factory = MLLMModelFactory()
    
    # Add open-source models
    try:
        yi_model = factory.create_model("yi-vl")
        yi_model.load_model()
        tester.add_model(yi_model)
        
        qwen_model = factory.create_model("qwen2-vl")
        qwen_model.load_model()
        tester.add_model(qwen_model)
        
    except Exception as e:
        logger.error(f"Failed to load open-source models: {e}")
    
    # Add closed-source models (requires API keys)
    try:
        # You'll need to set these environment variables
        import os
        openai_key = os.getenv("OPENAI_API_KEY")
        anthropic_key = os.getenv("ANTHROPIC_API_KEY")
        google_key = os.getenv("GOOGLE_API_KEY")
        
        if openai_key:
            gpt4o = factory.create_model("gpt-4o", api_key=openai_key)
            gpt4o.load_model()
            tester.add_model(gpt4o)
            
        if anthropic_key:
            claude = factory.create_model("claude-3.5-sonnet", api_key=anthropic_key)
            claude.load_model()
            tester.add_model(claude)
            
    except Exception as e:
        logger.error(f"Failed to load closed-source models: {e}")
    
    # Example dataset (you'll load this from your BESESR platform)
    sample_datasets = {
        "ASAP-AES": [
            {
                "essay_text": "The use of technology in education has revolutionized learning...",
                "prompt": "Write an essay about technology in education",
                "images": []
            }
            # Add more essays...
        ]
    }
    
    # Run evaluation
    results = tester.run_comprehensive_evaluation(sample_datasets)
    
    # Generate leaderboard
    leaderboard = tester.generate_leaderboard(results)
    
    print("=== BESESR Model Evaluation Results ===")
    for rank, (model, score) in enumerate(leaderboard.items(), 1):
        print(f"{rank}. {model}: {score:.3f}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()