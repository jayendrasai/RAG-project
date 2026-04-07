import os
import logging
import uuid
from pathlib import Path
from threading import Thread

from fastapi import APIRouter, UploadFile, File, Depends, HTTPException
from sqlalchemy.orm import Session

from backend.database import get_db
from backend.config import settings
from backend.models import Document, User
from backend.routes.auth import get_current_user

log = logging.getLogger(__name__)
router = APIRouter()

ALLOWED_TYPES = {"pdf", "png", "jpg", "jpeg", "txt"}


@router.post("/upload")
async def upload_file(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    ext = file.filename.rsplit(".", 1)[-1].lower() if "." in file.filename else ""
    if ext not in ALLOWED_TYPES:
        raise HTTPException(400, f"Unsupported file type: {ext}. Allowed: {', '.join(ALLOWED_TYPES)}")

    file_id = str(uuid.uuid4())
    filename = f"{file_id}_{file.filename}"
    file_path = os.path.join(settings.upload_dir, filename)

    content = await file.read()
    os.makedirs(settings.upload_dir, exist_ok=True)
    with open(file_path, "wb") as f:
        f.write(content)

    doc = Document(
        id=file_id,
        user_id=user.id,
        filename=file.filename,
        file_type=ext,
        file_path=file_path,
        file_size=len(content),
        status="pending",
    )
    db.add(doc)
    db.commit()

    thread = Thread(target=_run_ingestion_bg, args=(file_id, file_path, ext, str(user.id)))
    thread.start()

    log.info(f"Uploaded {file.filename} ({len(content)} bytes) -> {file_id} [user={user.id}]")

    return {
        "document_id": file_id,
        "filename": file.filename,
        "file_type": ext,
        "status": "pending",
        "message": "File uploaded, processing started",
    }


def _run_ingestion_bg(doc_id: str, file_path: str, file_type: str, user_id: str):
    from backend.database import SessionLocal
    db = SessionLocal()
    try:
        from backend.ingestion.pipeline import run_ingestion
        run_ingestion(doc_id, file_path, file_type, db, user_id=user_id)
    except Exception as e:
        log.exception(f"Background ingestion failed: {e}")
    finally:
        db.close()
