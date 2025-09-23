# BESESR — Benchmark for Essay Scoring Evaluation & Research

BESESR is a FastAPI-based platform to **evaluate automatic essay scoring models** across a curated set of datasets.  
Researchers download dataset CSVs, upload their model outputs, and get standardized metrics (QWK, Pearson, MAE, RMSE, F1, Accuracy) and a **live leaderboard**.

---

## ✨ Features

- Dynamic dataset discovery from a Hugging Face profile (with static fallback)
- CSV upload with validation and instant scoring
- Standard metrics: **QWK**, **Pearson**, **MAE**, **RMSE**, **F1**, **Accuracy**
- Leaderboard + platform stats
- Typed API responses via **Pydantic models** (`app/models/pydantic_models.py`)
- Simple vanilla JS frontend (`app/frontend/`)

---

## 🧱 Tech Stack

- **FastAPI**, **Uvicorn**
- **Pandas**, **NumPy**, **scikit-learn**
- **datasets**, **huggingface_hub**
- **SQLite** (auto-created) + **SQLAlchemy**
- **Pydantic v2** models for response schemas

---

## 🚀 Quickstart

### 1) Clone
```bash
git clone https://github.com/nlpatunt/besesr.git
cd besesr
```

### 2) Create & activate a virtual env
**Conda**
```bash
conda create -n besesr python=3.10 -y
conda activate besesr
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

### 4) (Optional) Hugging Face token
Create a `.env` file in the repo root:
```
HUGGINGFACE_TOKEN=hf_xxx_your_token_here
```
> Without this, the app falls back to a small static dataset configuration.

### 5) Run the server
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

- App UI: http://localhost:8000  
- API docs: http://localhost:8000/docs

> The SQLite DB is created automatically (via app code).  
> If you hit DB state issues locally, stop the server, delete `app/app.db`, start again.

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

**SQLite locked / weird DB state**  
- Stop server → delete `app/app.db` → start again

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

## 📜 License

MIT (update if needed).

---

## 🙏 Acknowledgments

Thanks to dataset authors and the open-source community (FastAPI, Hugging Face, etc.) that make BESESR possible.

