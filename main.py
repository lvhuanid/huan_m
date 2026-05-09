import requests
import json

# 直接使用 Ollama 原生 API (非 OpenAI 格式)
url = "http://localhost:11434/api/generate"

data = {
    "model": "qwen2.5:3b",
    "prompt": "你好，请介绍你自己。",
    "stream": False,
    "options": {
        "num_ctx": 2048
    }
}

response = requests.post(url, json=data)
result = response.json()
print("Ollama 返回的完整数据（包含 Prompt 处理信息）：")
print(json.dumps(result, indent=2, ensure_ascii=False))