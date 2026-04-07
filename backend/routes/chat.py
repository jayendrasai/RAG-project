import logging
import json
import uuid
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from backend.database import get_db
from backend.models import ChatSession, ChatMessage, RetrievalLog, DocumentChunk, User
from backend.retrieval.search import search_chunks
from backend.retrieval.generator import generate_answer
from backend.routes.auth import get_current_user

log = logging.getLogger(__name__)
router = APIRouter()


class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None


@router.post("/chat")
def chat(
    req: ChatRequest,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    # get or create session
    session = None
    if req.session_id:
        session = (
            db.query(ChatSession)
            .filter(ChatSession.id == req.session_id, ChatSession.user_id == user.id)
            .first()
        )

    if not session:
        session = ChatSession(title=req.message[:60], user_id=user.id)
        db.add(session)
        db.flush()

    # save user message
    user_msg = ChatMessage(
        session_id=session.id,
        role="user",
        content=req.message,
    )
    db.add(user_msg)
    db.flush()

    # retrieve relevant chunks (scoped to this user)
    chunks = search_chunks(req.message, user_id=str(user.id))

    # log retrievals
    for rank, chunk in enumerate(chunks):
        log_entry = RetrievalLog(
            message_id=user_msg.id,
            chunk_id=_find_chunk_id(db, chunk["embedding_id"]),
            document_id=chunk["metadata"].get("document_id"),
            query=req.message,
            similarity_score=chunk["score"],
            rank=rank + 1,
        )
        db.add(log_entry)

    # load recent chat history for context
    history = []
    prev_messages = (
        db.query(ChatMessage)
        .filter(ChatMessage.session_id == session.id)
        .order_by(ChatMessage.created_at)
        .all()
    )
    for m in prev_messages[-6:]:
        history.append({"role": m.role, "content": m.content})

    images = _collect_images(chunks)

    result = generate_answer(req.message, chunks, chat_history=history, images=images)

    assistant_msg = ChatMessage(
        session_id=session.id,
        role="assistant",
        content=result["answer"],
        sources=json.dumps(result["sources"]),
    )
    db.add(assistant_msg)

    if len(prev_messages) <= 1:
        session.title = req.message[:60]

    db.commit()

    return {
        "session_id": session.id,
        "message_id": assistant_msg.id,
        "answer": result["answer"],
        "sources": result["sources"],
    }


@router.get("/sessions")
def list_sessions(
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    sessions = (
        db.query(ChatSession)
        .filter(ChatSession.user_id == user.id)
        .order_by(ChatSession.updated_at.desc())
        .limit(50)
        .all()
    )
    return [
        {
            "id": s.id,
            "title": s.title,
            "created_at": s.created_at.isoformat(),
            "updated_at": s.updated_at.isoformat() if s.updated_at else None,
        }
        for s in sessions
    ]


@router.get("/sessions/{session_id}/messages")
def get_messages(session_id: str, db: Session = Depends(get_db)):
    session = db.query(ChatSession).filter(ChatSession.id == session_id).first()
    if not session:
        raise HTTPException(404, "Session not found")

    messages = (
        db.query(ChatMessage)
        .filter(ChatMessage.session_id == session_id)
        .order_by(ChatMessage.created_at)
        .all()
    )

    return {
        "session_id": session.id,
        "title": session.title,
        "messages": [
            {
                "id": m.id,
                "role": m.role,
                "content": m.content,
                "sources": json.loads(m.sources) if m.sources else None,
                "created_at": m.created_at.isoformat(),
            }
            for m in messages
        ],
    }


def _find_chunk_id(db: Session, embedding_id: str) -> Optional[str]:
    chunk = db.query(DocumentChunk).filter(DocumentChunk.embedding_id == embedding_id).first()
    return chunk.id if chunk else None


def _collect_images(chunks):
    return []
