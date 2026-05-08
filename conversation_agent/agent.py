# agent.py
import httpx
from config import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, CHAT_MODEL
from memory import ConversationMemory
from redis_store import save_session, load_session

async def call_llm(messages: list[dict]) -> str:
    async with httpx.AsyncClient() as client:
        resp = await client.post(
            f"{DEEPSEEK_BASE_URL}/chat/completions",
            headers={"Authorization": f"Bearer {DEEPSEEK_API_KEY}"},
            json={"model": CHAT_MODEL, "messages": messages, "temperature": 0.7},
            timeout=60.0
        )
        if resp.status_code != 200:
            raise Exception(f"API 错误 {resp.status_code}: {resp.text}")
        data = resp.json()
        if "choices" not in data:
            raise Exception(f"响应缺少 choices: {data}")
        return data["choices"][0]["message"]["content"].strip()
    
    
async def run_agent(session_id: str):
    # 尝试断线恢复
    saved = await load_session(session_id)
    if saved:
        mem = ConversationMemory.from_dict(saved)
        print(f"🔄 已恢复会话 {session_id}")
    else:
        mem = ConversationMemory()
        print(f"✨ 新建会话 {session_id}")

    print("Agent 启动，输入 'exit' 结束对话。")
    while True:
        try:
            user_input = input("👤 你: ")
        except (EOFError, KeyboardInterrupt):
            break
        if user_input.lower().strip() == "exit":
            break

        mem.add_message("user", user_input)
        # 压缩检查
        compressed = await mem.check_and_compress()
        if compressed:
            print("⚡ 上下文已自动压缩。")

        # 获取完整上下文并调用 LLM
        context = mem.get_full_context()
        reply = await call_llm(context)
        print(f"🤖 助手: {reply}")

        mem.add_message("assistant", reply)
        # 每次交互后持久化
        await save_session(session_id, mem.to_dict())
        print("💾 状态已保存。")