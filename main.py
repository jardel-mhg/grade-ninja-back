import time
import logging

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from s3_sync import download_all
from database import init_db
from routes import train, sessions, rows

logger = logging.getLogger("grade-ninja")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

app = FastAPI(
    title="Grade Ninja API",
    version="0.1.0",
    description="ML-powered leather grading API. Classifies industrial leather hides based on defect analysis.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def log_request_timing(request: Request, call_next):
    start = time.time()
    method = request.method
    path = request.url.path
    logger.info(f"→ {method} {path} received")
    response = await call_next(request)
    elapsed_ms = (time.time() - start) * 1000
    logger.info(f"← {method} {path} {response.status_code} ({elapsed_ms:.0f}ms)")
    return response

app.include_router(train.router)
app.include_router(sessions.router)
app.include_router(rows.router)

download_all()
init_db()


@app.get("/", tags=["health"], summary="Health check")
def health_check():
    """Returns service status. Use this to verify the API is running."""
    return {"status": "ok", "service": "grade-ninja-api"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
