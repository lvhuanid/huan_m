import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

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

model = os.getenv("LLM_MODEL_ID")

response = client.chat.completions.create(
    model=model,
    messages=[
        {"role": "system", "content": "你是一个乐于助人的助手"},
        {"role": "user", "content": "你好，请用一句话介绍你自己"}
    ],
    temperature=0.7,
    max_tokens=100
)

# temperature=0 → 输出严谨、逻辑化
# temperature=1.5 → 更发散，可能会有比喻或脑洞
# max_tokens=50 → 话说一半就被截断

print(response.choices[0].message.content)

messages = [{"role": "system", "content": "你是儿童科普老师，回答要简单有趣"}]
while True:
    user_input = input("你: ")
    messages.append({"role": "user", "content": user_input})
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.8
    )
    assistant_reply = response.choices[0].message.content
    print(f"AI: {assistant_reply}")
    messages.append({"role": "assistant", "content": assistant_reply})
