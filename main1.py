import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
    api_key=os.getenv("MODELSCOPE_API_KEY"),
    base_url=os.getenv("LLM_BASE_URL")
)

response = client.chat.completions.create(
    model=os.getenv("LLM_MODEL_ID"),
    messages=[
        {"role": "system", "content": "你是一个乐于助人的助手"},
        {"role": "user", "content": "你好，请用一句话介绍你自己"}
    ],
    temperature=0.7,
    max_tokens=100
)

print(response.choices[0].message.content)