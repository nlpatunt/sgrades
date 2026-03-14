<!-- # S-GRADES — Studying Generalization of Student Response Assessments in Diverse Evaluative Settings -->
![S-GRADES logo](assets/sgrades.svg)

S-GRADES is a FastAPI-based platform to **evaluate automatic essay scoring models** across a curated set of datasets.  
Researchers download dataset CSVs, upload their model outputs, and get standardized metrics (QWK, Pearson, MAE, RMSE, F1, Accuracy) and a **live leaderboard**.

---

## Paper

📄 **S-GRADES: Studying Generalization of Student Response Assessments in Diverse Evaluative Settings**
Proceedings of the 15th International Conference on Language Resources and Evaluation (LREC 2026)

Paper link: *(https://arxiv.org/abs/2603.10233)*

---

## Dataset Access

The datasets associated with this benchmark are publicly available on Hugging Face:

**S-GRADES Benchmark Datasets:** [[https://huggingface.co/nlpatunt](https://huggingface.co/collections/nlpatunt/automatic-grading-datasets-without-labels)]([https://huggingface.co/nlpatunt](https://huggingface.co/collections/nlpatunt/automatic-grading-datasets-without-labels))

The leaderboard and submission platform are available at: [https://sgrades.eng.unt.edu](https://sgrades.eng.unt.edu)

This benchmark is released for academic and research purposes and is designed to facilitate reproducible experimentation in automated essay scoring (AES) and automatic short answer grading (ASAG).


## ✨ Features

- Dynamic dataset discovery from a Hugging Face organization profile (with static fallback)
- CSV upload with validation, security scanning, and instant scoring
- Metric-aware evaluation split:
  - **Regression** (AES/ASAG scoring): **QWK**, **Pearson**, **MAE**, **RMSE**, **Accuracy**
  - **Classification** (binary/multi-class): **F1**, **Precision**, **Recall**, **Accuracy**
- N/A displayed for non-applicable metrics per dataset type
- Gold-Reference baseline for perfect-score comparison
- Leaderboard with sorting, filtering, and complete/partial benchmark toggle
- Platform stats dashboard
- Typed API responses via **Pydantic v2 models** (`app/models/pydantic_models.py`)
- PostgreSQL backend with SQLAlchemy ORM
- Vanilla JS frontend served via Apache (`app/frontend/`)
---

## 🧱 Tech Stack

- **Backend:** FastAPI, Uvicorn
- **Web Server:** Apache (reverse proxy)
- **Data Processing:** Pandas, NumPy, scikit-learn
- **Datasets:** `datasets`, `huggingface_hub`
- **Database:** SQLAlchemy + PostgreSQL
- **API Schemas:** Pydantic v2
- **Frontend:** HTML, CSS, JavaScript

## 🚀 Quickstart

### 1) Clone

```bash
git clone https://github.com/nlpatunt/sgrades.git
cd sgrades
```

### 2) Create & activate a virtual env

**Conda**

```bash
conda create -n sgrades python=3.10 -y
conda activate sgrades
```

**OR venv**

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate
```

### 3) Install dependencies

If `requirements.txt` exists:

```bash
pip install -r requirements.txt
```

Otherwise:

```bash
pip install fastapi "uvicorn[standard]" pandas numpy scikit-learn             datasets huggingface_hub python-dotenv SQLAlchemy
```

### 4) Configure environment variables

Create a `.env` file in the repo root:

#### Database Configuration

The application supports multiple database types via the `DB_TYPE` environment variable:

**Option 1: SQLite (default, create-on-boot)**

```env
DB_TYPE=create-on-boot
DB_PATH=./sgrades.db  # optional, defaults to ./sgrades.db
```

**Option 2: Local PostgreSQL**

```env
DB_TYPE=local-postgres
DB_USER=postgres          # optional, defaults to postgres
DB_PASSWORD=postgres      # optional, defaults to postgres
DB_HOST=localhost         # optional, defaults to localhost
DB_PORT=5432              # optional, defaults to 5432
DB_NAME=sgrades           # optional, defaults to sgrades
```

**Option 3: Local MySQL**

```env
DB_TYPE=local-mysql
DB_USER=root              # optional, defaults to root
DB_PASSWORD=root          # optional, defaults to root
DB_HOST=localhost         # optional, defaults to localhost
DB_PORT=3306              # optional, defaults to 3306
DB_NAME=sgrades           # optional, defaults to sgrades
```

**Option 4: Global PostgreSQL (e.g., Supabase)**

```env
DB_TYPE=global-postgres
# Either provide full connection string:
DATABASE_URL=postgresql://user:password@host:port/database
# Or provide individual components:
DB_USER=your_user
DB_PASSWORD=your_password
DB_HOST=your_host
DB_PORT=5432
DB_NAME=your_database
```

**Option 5: Global MySQL**

```env
DB_TYPE=global-mysql
# Either provide full connection string:
DATABASE_URL=mysql+pymysql://user:password@host:port/database
# Or provide individual components:
DB_USER=your_user
DB_PASSWORD=your_password
DB_HOST=your_host
DB_PORT=3306
DB_NAME=your_database
```

#### Hugging Face Token (Optional)

```env
HUGGINGFACE_TOKEN=hf_xxx_your_token_here
```

> Without this, the app falls back to a small static dataset configuration.

### 5) Run the server

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

- App UI: <http://localhost:8000>
- API docs: <http://localhost:8000/docs>

> **Database Notes:**
>
> - SQLite DB is created automatically on boot
> - For PostgreSQL/MySQL, ensure the database exists before starting
> - Database tables are created automatically on first run

---

## 📦 Project Structure

```
app/
  api/
    routes/
      datasets.py
      leaderboard.py
      output_submissions.py
  frontend/
    index.html
    css/style.css
    js/app.js
  models/
    database.py
    pydantic_models.py
  services/
    dataset_loader.py
    database_service.py
  db.py
  main.py
```

---

## 📚 Datasets

- On startup, datasets are auto-discovered from a Hugging Face user (see `services/dataset_loader.py`).
- If discovery fails or no token is provided, a **static fallback** is used.

List available datasets:

```
GET /api/datasets/
```

Download (if enabled in your build):

```
GET /api/datasets/download/all
GET /api/datasets/download/{dataset_name}
```

Dataset sample preview:

```
GET /api/datasets/{dataset_name}/sample?size=3
```

---

## 📤 Submitting Results

### CSV format (required)

```
essay_id,predicted_score
ASAP-AES_001,3.5
ASAP-AES_002,4.2
```

### Validate CSV

```
POST /submissions/validate-csv
form-data: file=<your.csv>
```

### Upload single dataset results

```
POST /submissions/upload-single-result
form-data:
  model_name          (text, required)
  dataset_name        (text, required)
  submitter_name      (text, required)
  submitter_email     (text, required)
  result_file         (file .csv, required)
  model_description   (text, optional)
```

### Check submission status

```
GET /submissions/submission-status/{submission_id}
```

---

## 🏆 Leaderboard & Stats

Leaderboard (metric is optional; defaults to QWK):

```
GET /api/leaderboard/?metric=avg_quadratic_weighted_kappa&limit=20
```

Platform stats (for homepage counters):

```
GET /api/leaderboard/stats
```

---

## 🧾 Typed Responses (Pydantic)

Core response schemas live in `app/models/pydantic_models.py`:

- Datasets: `DatasetInfo`, `DatasetsListResponse`, `DatasetSample`, etc.
- Submissions: `BenchmarkSubmissionResponse`, `SingleTestResponse`, `CSVValidationResponse`, `SubmissionStatus`
- Leaderboard: `LeaderboardEntry`, `CompleteLeaderboardEntry`, `LeaderboardResponse`
- Platform/Health: `PlatformStats`, `HealthCheck`, etc.

Typed responses keep the API consistent and consumable by the frontend and external users.

---

## 🔧 Troubleshooting

**Dataset dropdown empty in UI**  

- Open browser console → check errors  
- Visit `GET /api/datasets/` directly  
- Set `HUGGINGFACE_TOKEN` in `.env` and restart the server if empty

**Database connection issues**

- **SQLite locked**: Stop server → delete the `.db` file → restart
- **PostgreSQL/MySQL connection failed**: Verify credentials in `.env` and ensure database server is running
- **Database doesn't exist**: Create the database manually before starting the application

**Port in use**  

- Change port: `uvicorn app.main:app --port 8080`

**CORS**  

- Keep frontend & API on same origin (default)

---

## 🌐 Run on Another Computer

On the target machine, follow **Quickstart**.  
Run with `--host 0.0.0.0` and open from another device at `http://SERVER_IP:8000`.  
Ensure port **8000/TCP** is allowed by the OS/firewall.

---

## 🔁 Dev Workflow

```bash
# create a branch
git checkout -b feat/my-change

# run locally
uvicorn app.main:app --reload

# commit & push
git add -A
git commit -m "feat: describe your change"
git push origin feat/my-change
```

Open a PR to `main`.

---

## Related
- S-GRADES Platform: [S-GRADES](https://sgrades.eng.unt.edu/)
- Paper: *S-GRADES: Studying Generalization of Student Response Assessments in Diverse Evaluative Settings* — LREC 2026
  - arXiv: https://arxiv.org/abs/2603.10233
  - DOI: https://doi.org/10.48550/arXiv.2603.10233

## Citation
If you use this code or benchmark, please cite:
```bibtex
@article{seuti2026sgrades,
  title={S-GRADES: Studying Generalization of Student Response Assessments in Diverse Evaluative Settings},
  author={Seuti, Tasfia and Ray Choudhury, Sagnik},
  journal={arXiv preprint arXiv:2603.10233},
  year={2026}
}
```



---

## 🙏 Acknowledgments

Thanks to dataset authors and the open-source community (FastAPI, Hugging Face, etc.) that make S-GRADES possible.
