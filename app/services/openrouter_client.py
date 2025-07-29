import os
import requests
import asyncio
import json
from typing import Dict, Any

class OpenRouterClient:
    def __init__(self):
        self.api_key = os.getenv("OPENROUTER_API_KEY")
        self.base_url = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "http://localhost:8000",
            "X-Title": "BESESR Essay Grading API"
        }
    
    async def test_connection(self) -> Dict[str, Any]:
        """Test if OpenRouter API is accessible"""
        if not self.api_key:
            return {"status": "error", "message": "No API key found"}
        
        try:
            response = requests.get(
                f"{self.base_url}/models",
                headers=self.headers,
                timeout=10
            )
            if response.status_code == 200:
                return {"status": "connected", "message": "OpenRouter API working", "models_available": len(response.json()["data"])}
            else:
                return {"status": "error", "message": f"API returned {response.status_code}"}
        except Exception as e:
            return {"status": "error", "message": f"Connection failed: {str(e)}"}
    
    async def grade_essay(self, essay_text: str, dataset_name: str, prompt: str = None) -> Dict[str, Any]:
        """Grade an essay using OpenRouter API"""
        
        # Create grading prompt based on dataset
        grading_prompt = self._create_grading_prompt(essay_text, dataset_name, prompt)
        
        # Prepare API request
        payload = {
            "model": "anthropic/claude-3-haiku",  # Fast and cost-effective
            "messages": [
                {
                    "role": "system",
                    "content": "You are an expert essay grader. Provide scores and detailed feedback in JSON format."
                },
                {
                    "role": "user", 
                    "content": grading_prompt
                }
            ],
            "max_tokens": 1000,
            "temperature": 0.3
        }
        
        try:
            # Make async request to OpenRouter
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: requests.post(
                    f"{self.base_url}/chat/completions",
                    headers=self.headers,
                    json=payload,
                    timeout=30
                )
            )
            
            if response.status_code == 200:
                result = response.json()
                ai_response = result["choices"][0]["message"]["content"]
                
                # Parse AI response (we'll make this more robust)
                return self._parse_grading_response(ai_response, dataset_name)
            else:
                return {
                    "score": 0.0,
                    "feedback": f"API Error: {response.status_code}",
                    "model_used": "error"
                }
                
        except Exception as e:
            return {
                "score": 0.0,
                "feedback": f"Grading failed: {str(e)}",
                "model_used": "error"
            }
    
    def _create_grading_prompt(self, essay_text: str, dataset_name: str, prompt: str) -> str:
        """Create appropriate grading prompt based on dataset"""
        
        base_prompt = f"""
        Grade this essay for the {dataset_name} dataset.
        
        Essay Prompt: {prompt or "No specific prompt provided"}
        
        Essay to Grade:
        {essay_text}
        
        Please provide:
        1. A numerical score (appropriate for {dataset_name})
        2. Detailed feedback explaining the score
        
        Respond in JSON format:
        {{
            "score": <numerical_score>,
            "feedback": "<detailed explanation>"
        }}
        """
        
        return base_prompt
    
    def _parse_grading_response(self, ai_response: str, dataset_name: str) -> Dict[str, Any]:
        """Parse AI response into structured format"""
        try:
            # Try to extract JSON from response
            if "{" in ai_response and "}" in ai_response:
                start = ai_response.find("{")
                end = ai_response.rfind("}") + 1
                json_str = ai_response[start:end]
                parsed = json.loads(json_str)
                
                return {
                    "score": float(parsed.get("score", 0.0)),
                    "feedback": parsed.get("feedback", "No feedback provided"),
                    "model_used": "anthropic/claude-3-haiku"
                }
            else:
                # Fallback if JSON parsing fails
                return {
                    "score": 3.0,  # Default middle score
                    "feedback": ai_response,
                    "model_used": "anthropic/claude-3-haiku"
                }
        except:
            return {
                "score": 0.0,
                "feedback": f"Could not parse response: {ai_response}",
                "model_used": "anthropic/claude-3-haiku"
            }