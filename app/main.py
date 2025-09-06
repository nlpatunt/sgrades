from fastapi import FastAPI
from dotenv import load_dotenv
import os
from pathlib import Path
from app.models.pydantic_models import HealthCheck
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime

load_dotenv()

from app.api.routes import datasets, leaderboard
from app.api.routes import output_submissions
from app.config.database import init_database
from app.services.database_service import DatabaseService
from app.models.pydantic_models import HealthCheck, DatasetsListResponse, DatasetInfo

async def startup_event():
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
        print("⚠️ Continuing without database initialization...")

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
    3. **Submit CSV results** using `/api/submissions/upload-single`
    4. **View leaderboard** with your model's performance
    
    Submit your model results for evaluation across 25 curated academic datasets.
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

app.include_router(datasets.router, prefix="/api", tags=["datasets"])
app.include_router(leaderboard.router, prefix="/api", tags=["leaderboard"])
app.include_router(output_submissions.router, prefix="/api/submissions", tags=["submissions"])

from fastapi.responses import RedirectResponse  # (you already import this above)


@app.get("/api/available-datasets", response_model=DatasetsListResponse)
async def get_available_datasets_direct():
    """Direct route for frontend compatibility"""
    try:
        from app.services.dataset_loader import dataset_manager
        datasets_config = dataset_manager.datasets_config
        
        # Only show D_ datasets
        public_datasets = {}
        for name, config in datasets_config.items():
            if name.startswith("D_") and config.get("dataset_type") == "unlabeled":
                public_datasets[name] = config
        
        from app.models.pydantic_models import DatasetInfo, DatasetsListResponse
        datasets_list = []
        for dataset_name, config in public_datasets.items():
            dataset_info = DatasetInfo(
                name=dataset_name,
                description=config["description"],
                huggingface_id=config["huggingface_id"],
                essay_column=config["essay_column"],
                score_column=config["score_column"],
                prompt_column=config.get("prompt_column", "prompt"),
                score_range=config["score_range"],
                split=config["split"],
                status="active"
            )
            datasets_list.append(dataset_info)
        
        return DatasetsListResponse(
            datasets=datasets_list,
            total_count=len(datasets_list),
            data_source="direct_d_datasets",
            last_updated=datetime.now().isoformat()
        )
        
    except Exception as e:
        return DatasetsListResponse(
            datasets=[],
            total_count=0,
            data_source="error",
            last_updated=datetime.now().isoformat()
        )

@app.get("/health", response_model=HealthCheck, tags=["health"])
async def health_check():
    """Health check endpoint with database and dataset status"""
    try:
        from app.services.dataset_loader import dataset_manager
        
        stats = DatabaseService.get_platform_stats()
        db_status = "connected"
        
        # Check dataset loading capability
        try:
            datasets_config = dataset_manager.datasets_config
            hf_auth = dataset_manager.hf_loader.authenticated
        except:
            datasets_config = {}
            hf_auth = False
        
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

# -------------------
# API Info Endpoint
# -------------------
@app.get("/api-info", tags=["info"])
async def api_info():
    """Information about the new API workflow"""
    return {
        "workflow": "CSV Results Submission",
        "description": "Download datasets, run your model locally, submit CSV results",
        "endpoints": {
            "download_all_datasets": "/api/datasets/download/all",
            "download_single_dataset": "/api/datasets/download/{dataset_name}",
            "submit_single_result": "/api/submissions/upload-single",
            "submit_multiple_results": "/api/submissions/upload-batch",
            "view_leaderboard": "/api/submissions/leaderboard",
            "get_submission_template": "/api/submissions/template",
            "validate_csv": "/api/submissions/validate-csv",
            "check_submission_status": "/api/submissions/{submission_id}"
        },
        "workflow_steps": [
            "1. Download datasets using /api/datasets/download/all",
            "2. Extract ZIP file and read README.md for instructions",
            "3. Run your model on each dataset CSV file",
            "4. Create results CSV files with required columns for each dataset",
            "5. Submit results using /api/submissions/upload-single for each dataset",
            "6. View your model's performance on the leaderboard"
        ],
        "csv_format_info": {
            "note": "Each dataset has different required columns",
            "get_format": "GET /api/submissions/format/{dataset_name}",
            "get_template": "GET /api/submissions/template",
            "validation": "POST /api/submissions/validate-csv"
        },
        "available_endpoints": [
            "POST /api/submissions/upload-single - Upload single CSV file",
            "POST /api/submissions/upload-batch - Upload multiple CSV files",
            "GET /api/submissions/template - Get CSV format template",
            "POST /api/submissions/validate-csv - Validate CSV before submission",
            "GET /api/submissions/{submission_id} - Check submission details",
            "GET /api/submissions/leaderboard - View rankings"
        ]
    }

# -------------------
# Startup Message
# -------------------
@app.on_event("startup")
async def startup_message():
    """Print startup message"""
    print("\n" + "="*60)
    print("🎉 BESESR Platform Started Successfully!")
    print("="*60)
    print("📍 Frontend: http://localhost:8000/")
    print("📖 API Docs: http://localhost:8000/docs")
    print("🔍 Health Check: http://localhost:8000/health")
    print("📊 Datasets: http://localhost:8000/api/available-datasets/")
    print("📤 Submissions: http://localhost:8000/api/submissions/template")
    print("🏆 Leaderboard: http://localhost:8000/api/submissions/leaderboard")
    print("="*60 + "\n")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)