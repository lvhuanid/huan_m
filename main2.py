import os
from openai import OpenAI
import json
from dotenv import load_dotenv

load_dotenv()

model = os.getenv("LLM_MODEL_ID")
client = OpenAI(
    api_key=os.getenv("MODELSCOPE_API_KEY"),
    base_url=os.getenv("LLM_BASE_URL")
)
# EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")
# ollama_client = OpenAI(
#     base_url="http://localhost:11434/v1",
#     api_key="ollama"
# )
#   """调用 Ollama 获取文本的向量表示"""
# response = ollama_client.embeddings.create(
#     model=EMBEDDING_MODEL,
#     input=text
# )
# resp =  ollama_client.embeddings.create(
#     model=EMBEDDING_MODEL,
#     input=["今天天气真好", "下雨了，好冷"]
# )

prompt = """
从以下文本中提取人名、公司和职位，以 JSON 格式返回。
只返回 JSON，不要任何解释。
文本：王小蒙在北京智源科技有限公司担任算法工程师。
输出格式示例：{"name": "...", "company": "...", "position": "..."}
"""

response = client.chat.completions.create(
    model=model,
     messages=[{"role": "user", "content": prompt}],
    temperature=0,
    response_format={"type": "json_object"}
)

content = response.choices[0].message.content
# 有时候模型会包在 ```json 代码块里，需要处理一下
if "```json" in content:
    content = content.split("```json")[1].split("```")[0].strip()
data = json.loads(content)
print(data)