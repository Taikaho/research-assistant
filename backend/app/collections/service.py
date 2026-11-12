import uuid
from collections.abc import Sequence

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.collections.schemas import CollectionCreate, CollectionUpdate
from app.database.models import Collection


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
