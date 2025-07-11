from fastapi import FastAPI
from dotenv import load_dotenv
import os
from app.services.openrouter_client import OpenRouterClient
from app.api.routes import datasets, essays, models, leaderboard  # ← Add models and leaderboard

# Load environment variables
load_dotenv()

# Create FastAPI app
app = FastAPI(
    title="BESESR - Essay Grading Benchmarking Platform",  # ← Updated title
    description="Benchmarking platform for automatic essay grading models. Submit your models for evaluation across 12 curated datasets.",  # ← Updated description
    version="1.0.0"
)

# Include all routes
app.include_router(datasets.router)    # Dataset information endpoints
app.include_router(essays.router)     # Internal testing endpoints  
app.include_router(models.router)     # Model submission and evaluation endpoints
app.include_router(leaderboard.router)  # Leaderboard and ranking endpoints

# Initialize OpenRouter client
openrouter_client = OpenRouterClient()

@app.get("/")
async def root():
    return {
        "message": "BESESR Essay Grading Benchmarking Platform",
        "description": "Submit your essay grading models for evaluation across 12 curated datasets",
        "endpoints": {
            "datasets": "/datasets/ - View available datasets",
            "submit_model": "/models/submit - Submit your model for evaluation", 
            "leaderboard": "/leaderboard/ - View model rankings",
            "documentation": "/docs - Full API documentation"
        },
        "total_datasets": 12,
        "supported_metrics": ["QWK", "Pearson", "F1", "Accuracy", "MAE", "RMSE"]
    }

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "BESESR Benchmarking Platform"}

@app.get("/test-openrouter")
async def test_openrouter():
    result = await openrouter_client.test_connection()
    return result