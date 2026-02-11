# Grade Ninja API

Backend API for the Grade Ninja leather grading system. Classifies industrial leather hides into grades (B/C/D/E/EN) based on defect analysis.

## Quick Start

```bash
pip install -r requirements.txt
python3 main.py
```

Server starts at `http://localhost:8000`

## API Docs

- **Swagger UI:** http://localhost:8000/docs
- **ReDoc:** http://localhost:8000/redoc

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | Health check |
| GET | `/api/projects` | List all projects |
| GET | `/api/projects/{id}` | Get project by ID |
| POST | `/api/projects` | Create project |
| PUT | `/api/projects/{id}` | Update project |
| GET | `/api/datasets` | List all datasets |
| GET | `/api/datasets/{id}` | Get dataset by ID |
| POST | `/api/datasets` | Create dataset |
| POST | `/api/train` | Start training job |
| GET | `/api/train/{job_id}` | Get training status |
| POST | `/api/inference` | Predict hide grade |

## Deployment

Deployed on **AWS App Runner** via `apprunner.yaml`. Auto-deploys on push to `main`.

## Tech Stack

- **Framework:** FastAPI
- **Runtime:** Python 3.12
- **Hosting:** AWS App Runner
