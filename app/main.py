from fastapi import FastAPI
from dotenv import load_dotenv
import os
from pathlib import Path

from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse
from fastapi.middleware.cors import CORSMiddleware

# Import all route modules
from app.services.openrouter_client import OpenRouterClient
from app.api.routes import datasets, essays, leaderboard
from app.api.routes.output_submissions import router as output_router
from app.config.database import init_database
from app.services.database_service import DatabaseService

# -------------------
# Load env vars
# -------------------
load_dotenv()

# -------------------
# Startup task
# -------------------
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
    allow_origins=["*"],   # tighten in prod
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

# Essays management
app.include_router(essays.router, prefix="/api")

# Leaderboard and results
app.include_router(leaderboard.router, prefix="/api")

# CSV Results submission (new workflow)
app.include_router(output_router, prefix="/api/outputs")

# -------------------
# Services
# -------------------
openrouter_client = OpenRouterClient()

# -------------------
# Health & Debug endpoints
# -------------------
@app.get("/health")
async def health_check():
    """Health check endpoint with database and dataset status"""
    try:
        from app.services.dataset_loader import dataset_manager
        
        stats = DatabaseService.get_platform_stats()
        db_status = "healthy"
        
        # Check dataset loading capability
        datasets_config = dataset_manager.datasets_config
        hf_auth = dataset_manager.hf_loader.authenticated
        
        dataset_status = {
            "total_configured": len(datasets_config),
            "huggingface_authenticated": hf_auth,
            "sample_datasets": list(datasets_config.keys())[:3]
        }
        
    except Exception as e:
        db_status = f"error: {str(e)}"
        dataset_status = {"error": str(e)}

    return {
        "status": "healthy",
        "service": "BESESR Benchmarking Platform",
        "database": db_status,
        "datasets": dataset_status,
        "workflow": "csv_submission",
        "version": "1.0.0"
    }

@app.get("/test-openrouter")
async def test_openrouter():
    """Test OpenRouter connection (legacy)"""
    return await openrouter_client.test_connection()

@app.get("/debug/database")
async def debug_database():
    """Debug endpoint to check database status"""
    try:
        stats = DatabaseService.get_platform_stats()
        leaderboard_data = DatabaseService.get_leaderboard(limit=5)
        
        # Check database file
        db_file_exists = os.path.exists("besesr.db")
        db_size = os.path.getsize("besesr.db") if db_file_exists else 0
        
        return {
            "database_file": "besesr.db",
            "file_exists": db_file_exists,
            "file_size_mb": round(db_size / (1024 * 1024), 2),
            "stats": stats,
            "sample_leaderboard": leaderboard_data,
            "status": "connected"
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}

@app.get("/debug/datasets")
async def debug_datasets():
    """Debug endpoint to check dataset loading"""
    try:
        from app.services.dataset_loader import dataset_manager
        
        datasets_config = dataset_manager.datasets_config
        
        # Test loading one dataset
        test_dataset = list(datasets_config.keys())[0] if datasets_config else None
        test_sample = None
        
        if test_dataset:
            test_sample = dataset_manager.load_dataset_for_evaluation(test_dataset, sample_size=2)
        
        return {
            "total_datasets": len(datasets_config),
            "dataset_names": list(datasets_config.keys()),
            "huggingface_authenticated": dataset_manager.hf_loader.authenticated,
            "test_dataset": test_dataset,
            "test_sample_count": len(test_sample) if test_sample else 0,
            "test_sample_preview": test_sample[0] if test_sample else None,
            "status": "working" if test_sample else "not_working"
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}

@app.post("/debug/reset-database")
async def reset_database():
    """Reset database (development only)"""
    try:
        from app.config.database import reset_database
        reset_database()
        DatabaseService.initialize_datasets()
        return {"message": "Database reset successfully", "status": "completed"}
    except Exception as e:
        return {"message": "Database reset failed", "error": str(e), "status": "error"}

@app.get("/api-info")
async def api_info():
    """Information about the new API workflow"""
    return {
        "workflow": "CSV Results Submission",
        "description": "Download datasets, run your model locally, submit CSV results",
        "endpoints": {
            "download_all_datasets": "/api/datasets/download/all",
            "download_single_dataset": "/api/datasets/download/{dataset_name}",
            "submit_results": "/api/outputs/upload-results",
            "view_leaderboard": "/api/leaderboard/",
            "get_submission_template": "/api/outputs/template"
        },
        "workflow_steps": [
            "1. Download datasets using /api/datasets/download/all",
            "2. Extract ZIP file and read README.md for instructions",
            "3. Run your model on each dataset CSV file",
            "4. Create results CSV files with essay_id,predicted_score columns",
            "5. Submit all results using /api/outputs/upload-results",
            "6. View your model's performance on the leaderboard"
        ],
        "required_csv_format": {
            "columns": ["essay_id", "predicted_score"],
            "example": "essay_id,predicted_score\nASAP-AES_0,3.5\nASAP-AES_1,4.2"
        }
    }