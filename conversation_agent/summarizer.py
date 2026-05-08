# summarizer.py
import httpx
from config import OLLAMA_BASE_URL, SUMMARIZE_MODEL, DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL

async def summarize_conversation(messages: list[dict], use_local: bool = True) -> str:
    """
    对对话历史生成摘要。
    - use_local=True: 调用 Ollama 模型，适合低延迟、无成本场景。
    - use_local=False: 调用 DeepSeek API（生产推荐）。
    """
    # 构造 prompt，要求输出极简摘要，保留关键实体、决策、用户偏好等
    prompt = "请将以下对话历史压缩为一段200字以内的摘要，保留所有关键事实、用户选择、重要数字和任务进展：\n"
    for m in messages:
        prompt += f"{m['role']}: {m['content']}\n"

    if use_local:
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                f"{OLLAMA_BASE_URL}/api/generate",
                json={"model": SUMMARIZE_MODEL, "prompt": prompt, "stream": False},
                timeout=60.0
            )
            return resp.json()["response"].strip()
    else:
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                f"{DEEPSEEK_BASE_URL}/chat/completions",
                headers={"Authorization": f"Bearer {DEEPSEEK_API_KEY}"},
                json={
                    "model": "deepseek-chat",
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.3,
                    "max_tokens": 300
                },
                timeout=60.0
            )
            return resp.json()["choices"][0]["message"]["content"].strip()