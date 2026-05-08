# config.py
# docker run -d --name redis-6380 -p 6380:6379 redis:latest
import os
from dotenv import load_dotenv

load_dotenv()  # 从 .env 文件加载环境变量
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_BASE_URL = "https://api.deepseek.com"

OLLAMA_BASE_URL = "http://localhost:11434"
SUMMARIZE_MODEL = "llama3.2:1b"          # Ollama 模型，用于摘要（可更换为 deepseek-chat）
CHAT_MODEL = "deepseek-chat"
MAX_TOKEN_BUDGET = 6000               # 触发压缩的总 token 上限
RECENT_ROUNDS_KEPT = 4                # 压缩后保留的最近完整对话轮数（每轮 user+assistant）
REDIS_URL = "redis://localhost:6380/0"
SESSION_TTL = 3600 * 24               # Redis 中 session 过期时间（秒）