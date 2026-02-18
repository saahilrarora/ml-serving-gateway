import logging
from contextlib import asynccontextmanager  # lets us define startup/shutdown logic

from fastapi import FastAPI

# import route modules — each has its own APIRouter
from gateway.routes import health, predict

# configure logging so we see INFO-level messages in the terminal
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Runs once on startup (before yield) and once on shutdown (after yield).
    In Phase 2, we'll initialize the model registry here."""
    logger.info("ML Serving Gateway starting up")
    yield  # server is running and accepting requests between startup and shutdown
    logger.info("ML Serving Gateway shutting down")


# create the FastAPI application instance
app = FastAPI(
    title="ML Serving Gateway",
    description="Production-grade model serving with batching, routing, and optimization",
    version="0.1.0",
    lifespan=lifespan,  # wire up the startup/shutdown handler
)

# register route groups — this makes /health, /ready, and /predict available
app.include_router(health.router)
app.include_router(predict.router)
