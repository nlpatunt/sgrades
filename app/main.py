from fastapi import FastAPI
from dotenv import load_dotenv
import os
from pathlib import Path
from app.models.pydantic_models import HealthCheck
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime
from app.api.routes import datasets, leaderboard
from app.api.routes import output_submissions
from app.config.database import init_database
from app.services.database_service import DatabaseService

load_dotenv()

async def startup_event():
    """Initialize database and default data on startup"""
    print("🚀 Starting BESESR Platform...")
    print("🗄️ Initializing database...")
    try:
        init_database()
        print("✅ Database tables created successfully!")

        # Initialize default datasets only if not present
        DatabaseService.initialize_datasets()
        print("✅ Default datasets initialized!")

        print("🎉 BESESR Platform startup completed successfully!")
    except Exception as e:
        print(f"❌ Error during startup: {e}")
        raise e

# -------------------
# FastAPI app
# -------------------
app = FastAPI(
    title="BESESR - Essay Grading Benchmarking Platform",
    description="""
    Benchmarking platform for automatic essay grading models. 
    
    ## New Workflow:
    1. **Download datasets** using `/datasets/download/all`
    2. **Run your model** locally on all datasets
    3. **Submit CSV results** using `/outputs/upload-results`
    4. **View leaderboard** with your model's performance
    
    Submit your model results for evaluation across 12 curated academic datasets.
    """,
    version="1.0.0",
    on_startup=[startup_event]
)

# -------------------
# CORS
# -------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------
# Static frontend
# -------------------
FRONTEND_DIR = Path(__file__).parent / "frontend"  # app/frontend
app.mount("/frontend", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")

# Redirect root "/" to the frontend
@app.get("/", include_in_schema=False)
async def root_redirect():
    return RedirectResponse(url="/frontend/")

# -------------------
# API Routers
# -------------------
# Dataset management (with download functionality)
app.include_router(datasets.router, prefix="/api")

# Leaderboard and results
app.include_router(leaderboard.router, prefix="/api")

# CSV Results submission (new workflow)

app.include_router(output_submissions.router)

@app.get("/health", response_model=HealthCheck)
async def health_check():
    """Health check endpoint with database and dataset status"""
    try:
        from app.services.dataset_loader import dataset_manager
        
        stats = DatabaseService.get_platform_stats()
        db_status = "connected"
        
        # Check dataset loading capability
        datasets_config = dataset_manager.datasets_config
        hf_auth = dataset_manager.hf_loader.authenticated
        
        complete_benchmarks = len(DatabaseService.get_complete_benchmark_leaderboard(limit=1000))
        
        return HealthCheck(
            status="healthy",
            service="BESESR Benchmarking Platform",
            database_connection=db_status,
            timestamp=datetime.now().isoformat(),
            complete_benchmarks_available=complete_benchmarks,
            models_available=stats.get("total_models_submitted", 0),
            evaluations_completed=stats.get("total_evaluations_completed", 0)
        )
        
    except Exception as e:
        return HealthCheck(
            status="unhealthy",
            service="BESESR Benchmarking Platform",
            database_connection="error",
            timestamp=datetime.now().isoformat(),
            error=str(e)
        )

@app.get("/api-info")
async def api_info():
    """Information about the new API workflow"""
    return {
        "workflow": "CSV Results Submission",
        "description": "Download datasets, run your model locally, submit CSV results",
        "endpoints": {
            "download_all_datasets": "/api/datasets/download/all",
            "download_single_dataset": "/api/datasets/download/{dataset_name}",
            "submit_single_result": "/submissions/upload-single-result",  # ✅ Fixed
            "submit_multiple_results": "/submissions/upload-results",     # ✅ Fixed
            "view_leaderboard": "/api/leaderboard/",
            "get_submission_template": "/submissions/template",           # ✅ Fixed
            "validate_csv": "/submissions/validate-csv",                 # ✅ Added
            "check_submission_status": "/submissions/submission-status/{id}" # ✅ Added
        },
        "workflow_steps": [
            "1. Download datasets using /api/datasets/download/all",
            "2. Extract ZIP file and read README.md for instructions",
            "3. Run your model on each dataset CSV file",
            "4. Create results CSV files with essay_id,predicted_score columns",
            "5. Submit results using /submissions/upload-single-result or /submissions/upload-results",
            "6. View your model's performance on the leaderboard"
        ],
        "required_csv_format": {
            "columns": ["essay_id", "predicted_score"],
            "example": "essay_id,predicted_score\nASAP-AES_0,3.5\nASAP-AES_1,4.2"
        },
        "available_endpoints": [
            "POST /submissions/upload-single-result - Upload single CSV file",
            "POST /submissions/upload-results - Upload multiple CSV files",
            "GET /submissions/template - Get CSV format template",
            "POST /submissions/validate-csv - Validate CSV before submission",
            "GET /submissions/submission-status/{id} - Check submission status"
        ]
    }