from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from database import init_db
from routes import projects, datasets, train, inference, sessions, rows

app = FastAPI(
    title="Grade Ninja API",
    version="0.1.0",
    description=(
        "ML-powered leather grading API. "
        "Classifies industrial leather hides into grades (B/C/D/E/EN) based on defect analysis.\n\n"
        "**Note:** All endpoints currently return mocked data for frontend integration."
    ),
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(projects.router)
app.include_router(datasets.router)
app.include_router(train.router)
app.include_router(inference.router)
app.include_router(sessions.router)
app.include_router(rows.router)

init_db()


@app.get("/", tags=["health"], summary="Health check")
def health_check():
    """Returns service status. Use this to verify the API is running."""
    return {"status": "ok", "service": "grade-ninja-api"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
