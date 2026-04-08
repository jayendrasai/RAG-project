from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import logging

from backend.config import settings
from backend.database import init_db

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
log = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("Starting up — initializing DB tables")
    settings.ensure_upload_dir()
    init_db()
    yield
    log.info("Shutting down")


def create_app() -> FastAPI:
    app = FastAPI(
        title="Multimodal RAG",
        version="1.0.0",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    from backend.routes import upload, chat, documents, auth
    app.include_router(auth.router, prefix="/api", tags=["auth"])
    app.include_router(upload.router, prefix="/api", tags=["upload"])
    app.include_router(chat.router, prefix="/api", tags=["chat"])
    app.include_router(documents.router, prefix="/api", tags=["documents"])

    @app.get("/health")
    def health():
        return {"status": "ok"}

    return app


app = create_app()
