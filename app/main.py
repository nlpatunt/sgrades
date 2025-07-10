from fastapi import FastAPI
from dotenv import load_dotenv
import os
from app.services.openrouter_client import OpenRouterClient

# Load environment variables
load_dotenv()

# Create FastAPI app
app = FastAPI(
    title="BESESR - Essay Grading API",
    description="Automatic grading system for essays and long answers",
    version="1.0.0"
)

# Initialize OpenRouter client
openrouter_client = OpenRouterClient()

@app.get("/")
async def root():
    return {"message": "BESESR Essay Grading API is running!"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "BESESR API"}

@app.get("/test-openrouter")
async def test_openrouter():
    result = await openrouter_client.test_connection()
    return result