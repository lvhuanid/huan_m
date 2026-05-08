# redis_store.py
import json
import redis.asyncio as redis
from config import REDIS_URL, SESSION_TTL

async def get_redis():
    return await redis.from_url(REDIS_URL, decode_responses=True)

async def save_session(session_id: str, memory_dict: dict):
    r = await get_redis()
    key = f"agent:session:{session_id}"
    await r.set(key, json.dumps(memory_dict, ensure_ascii=False), ex=SESSION_TTL)

async def load_session(session_id: str) -> dict | None:
    r = await get_redis()
    key = f"agent:session:{session_id}"
    data = await r.get(key)
    return json.loads(data) if data else None

async def delete_session(session_id: str):
    r = await get_redis()
    await r.delete(f"agent:session:{session_id}")