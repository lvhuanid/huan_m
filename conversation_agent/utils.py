# utils.py
import tiktoken

ENCODER = tiktoken.get_encoding("cl100k_base")

def count_tokens(messages: list[dict]) -> int:
    """估算消息列表的 token 总数（简单按内容拼接计算，实际需计入 role 开销）"""
    total = 0
    for msg in messages:
        total += len(ENCODER.encode(msg.get("content", "")))
        total += 4  # 每条消息元数据（role等）的 token 近似补偿
    return total