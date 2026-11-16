import uuid
from collections.abc import Sequence
from typing import Any, cast

from sqlalchemy import delete, select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.engine import CursorResult
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.collections.schemas import CollectionCreate, CollectionUpdate
from app.database.models import Collection, CollectionDocument, Document


async def create_collection(db: AsyncSession, data: CollectionCreate) -> Collection:
    collection = Collection(name=data.name, description=data.description)
    db.add(collection)
    try:
        await db.commit()
        await db.refresh(collection)
    except IntegrityError:
        await db.rollback()
        raise
    return collection


async def list_collections(db: AsyncSession) -> Sequence[Collection]:
    result = await db.execute(select(Collection).order_by(Collection.created_at))
    return result.scalars().all()


async def get_collection(db: AsyncSession, collection_id: uuid.UUID) -> Collection | None:
    result = await db.execute(select(Collection).where(Collection.id == collection_id))
    return result.scalar_one_or_none()


async def update_collection(
    db: AsyncSession, collection: Collection, data: CollectionUpdate
) -> Collection:
    if data.name is not None:
        collection.name = data.name
    if data.description is not None:
        collection.description = data.description
    try:
        await db.commit()
        await db.refresh(collection)
    except IntegrityError:
        await db.rollback()
        raise
    return collection


async def delete_collection(db: AsyncSession, collection: Collection) -> None:
    await db.delete(collection)
    await db.commit()


# ---------------------------------------------------------------------------
# Collection membership
# ---------------------------------------------------------------------------


async def find_missing_collection_ids(
    db: AsyncSession, collection_ids: Sequence[uuid.UUID]
) -> list[uuid.UUID]:
    """Return the subset of collection_ids that do not exist, preserving order."""
    if not collection_ids:
        return []
    result = await db.execute(select(Collection.id).where(Collection.id.in_(collection_ids)))
    existing = set(result.scalars().all())
    return [cid for cid in dict.fromkeys(collection_ids) if cid not in existing]


async def link_document(
    db: AsyncSession, collection_ids: Sequence[uuid.UUID], document_id: uuid.UUID
) -> None:
    """Add a document to collections, ignoring links that already exist.

    Does not commit; the caller owns the transaction.
    """
    unique_ids = list(dict.fromkeys(collection_ids))
    if not unique_ids:
        return
    stmt = (
        pg_insert(CollectionDocument)
        .values([{"collection_id": cid, "document_id": document_id} for cid in unique_ids])
        .on_conflict_do_nothing(index_elements=["collection_id", "document_id"])
    )
    await db.execute(stmt)


async def add_document(
    db: AsyncSession, collection_id: uuid.UUID, document_id: uuid.UUID
) -> None:
    """Add a document to a collection. Idempotent."""
    await link_document(db, [collection_id], document_id)
    await db.commit()


async def remove_document(
    db: AsyncSession, collection_id: uuid.UUID, document_id: uuid.UUID
) -> bool:
    """Remove a document from a collection without deleting the document.

    Returns False if the document was not in the collection.
    """
    result = cast(
        CursorResult[Any],
        await db.execute(
            delete(CollectionDocument).where(
                CollectionDocument.collection_id == collection_id,
                CollectionDocument.document_id == document_id,
            )
        ),
    )
    await db.commit()
    return bool(result.rowcount)


async def list_documents(db: AsyncSession, collection_id: uuid.UUID) -> Sequence[Document]:
    """List the documents in a collection, in the order they were added."""
    result = await db.execute(
        select(Document)
        .join(CollectionDocument, CollectionDocument.document_id == Document.id)
        .where(CollectionDocument.collection_id == collection_id)
        .order_by(CollectionDocument.added_at, Document.created_at)
    )
    return result.scalars().all()
