# memory.py
from typing import Optional
from config import MAX_TOKEN_BUDGET, RECENT_ROUNDS_KEPT
from utils import count_tokens
from summarizer import summarize_conversation

class ConversationMemory:
    def __init__(self):
        self.summary: Optional[str] = None       # 历史摘要
        self.recent_messages: list[dict] = []    # 最近 RECENT_ROUNDS_KEPT 轮完整消息

    def add_message(self, role: str, content: str):
        self.recent_messages.append({"role": role, "content": content})

    def get_full_context(self) -> list[dict]:
        """拼接完整上下文供 LLM 调用"""
        context = []
        if self.summary:
            context.append({"role": "system", "content": f"对话历史摘要：{self.summary}"})
        context.extend(self.recent_messages)
        return context

    async def check_and_compress(self):
        """若总 token 超过阈值，则执行压缩"""
        if count_tokens(self.get_full_context()) < MAX_TOKEN_BUDGET:
            return False  # 无需压缩
        # 准备压缩的历史（摘要 + 最近消息之前的旧消息），但此处我们是将全部 recent 中除了最后保留轮次的部分放入摘要
        total_msgs = len(self.recent_messages)
        keep = RECENT_ROUNDS_KEPT * 2  # user+assistant 各一条为一轮
        if total_msgs <= keep:
            # 消息太少，无法压缩，直接对全部做摘要并清空 recent（极端情况）
            to_summarize = self.recent_messages
            self.recent_messages = []
        else:
            to_summarize = self.recent_messages[:-keep]
            self.recent_messages = self.recent_messages[-keep:]

        # 将旧消息连同已有摘要合并生成新摘要
        prior = f"先前摘要：{self.summary}\n" if self.summary else ""
        new_summary = await summarize_conversation(to_summarize, use_local=True)  # 可切换
        self.summary = (prior + new_summary)[:2000]  # 控制摘要长度
        return True

    def to_dict(self) -> dict:
        return {
            "summary": self.summary,
            "recent_messages": self.recent_messages
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ConversationMemory":
        mem = cls()
        mem.summary = data.get("summary")
        mem.recent_messages = data.get("recent_messages", [])
        return mem