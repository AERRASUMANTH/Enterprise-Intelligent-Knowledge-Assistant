"""FastAPI application entry point.

Run from the `backend/` folder:
    uvicorn app.main:app --reload --reload-dir app --port 8001
"""

import logging

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api import chat, documents, integrations, multi_agent, rag
from app.config import settings

# Show the app's own warnings (e.g. Gemini retries) in the server console.
logging.basicConfig(level=logging.WARNING, format="%(levelname)s:     %(name)s: %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(
    title=settings.app_name,
    description="Backend API for the Enterprise Intelligent Knowledge Assistant.",
    version="1.0.0",
)

# Allow the frontend (running on another port) to call this API from the browser.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in settings.cors_origins.split(",") if o.strip()],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(Exception)
async def unhandled_error(request: Request, exc: Exception) -> JSONResponse:
    """Return unexpected errors as JSON (details stay in the server log)."""
    logger.exception("Unhandled error on %s %s", request.method, request.url.path)
    return JSONResponse(status_code=500, content={"detail": "Internal server error. See server logs."})


app.include_router(documents.router)
app.include_router(rag.router)
app.include_router(chat.router)
app.include_router(multi_agent.router)
app.include_router(integrations.router)  # status only; connectors are not active


@app.get("/health", tags=["system"])
def health_check() -> dict:
    """Simple liveness check used to confirm the API is running."""
    return {
        "status": "ok",
        "app": settings.app_name,
        "environment": settings.environment,
        # Report only whether a key is set, never the key itself.
        "gemini_configured": settings.gemini_api_key is not None,
        "gemini_model": settings.gemini_model,
    }
