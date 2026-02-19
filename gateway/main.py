import logging
from contextlib import asynccontextmanager  # lets us define startup/shutdown logic

from fastapi import FastAPI

# import route modules — each has its own APIRouter
from gateway.routes import health, models, predict, routing
from gateway.registry.model_store import ModelStore  # in-memory model registry
from gateway.batching.batcher import AsyncBatcher     # adaptive request batcher
from gateway.batching.config import BatchConfig
from gateway.routing.router import TrafficRouter      # A/B traffic routing

# configure logging so we see INFO-level messages in the terminal
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Runs once on startup (before yield) and once on shutdown (after yield)."""
    # create the shared model store and attach it to app.state
    # so all route handlers can access it via request.app.state.model_store
    app.state.model_store = ModelStore()

    # start the batcher — it spawns a background task that drains queues
    app.state.batcher = AsyncBatcher(app.state.model_store, BatchConfig())
    await app.state.batcher.start()

    # create the traffic router for A/B routing
    app.state.router = TrafficRouter()

    logger.info("ML Serving Gateway starting up — model store + batcher + router initialized")
    yield  # server is running and accepting requests between startup and shutdown

    await app.state.batcher.stop()
    logger.info("ML Serving Gateway shutting down")


# create the FastAPI application instance
app = FastAPI(
    title="ML Serving Gateway",
    description="Production-grade model serving with batching, routing, and optimization",
    version="0.1.0",
    lifespan=lifespan,  # wire up the startup/shutdown handler
)

# register route groups — this makes /health, /ready, /predict, and /models available
app.include_router(health.router)
app.include_router(predict.router)
app.include_router(models.router)
app.include_router(routing.router)
