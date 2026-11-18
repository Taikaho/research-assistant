from fastapi import APIRouter

from app.chat.router import router as chat_router
from app.collections.router import router as collections_router
from app.documents.router import router as documents_router
from app.embeddings.router import router as embeddings_router
from app.llm.router import router as models_router
from app.retrieval.router import router as retrieval_router

api_router = APIRouter(prefix="/api/v1")
api_router.include_router(collections_router)
api_router.include_router(documents_router)
api_router.include_router(embeddings_router)
api_router.include_router(retrieval_router)
api_router.include_router(chat_router)
api_router.include_router(models_router)
