from openai import OpenAI

# ================= 配置区域（仅这里修改） =================
# 指向 Ollama 的本地兼容接口
client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama"  # 这里可以随便填，Ollama 不需要真实 Key
)

# ================= 1. 最小知识库 =================
knowledge_base = [
    "豆包是字节跳动于2024年发布的AI大模型。",
    "项目驱动学习法（Project-based Learning）的核心是：先做出来，再补理论。",
    "流式输出（Streaming）的原理是服务端一边生成数据，客户端一边接收。"
]

# ================= 2. 极简“检索器” =================
def find_relevant_info(user_question):
    return "\n".join(knowledge_base)

# ================= 3. 核心问答逻辑 =================
def chat_with_knowledge():
    user_input = input("请输入你的问题：")
    
    context = find_relevant_info(user_input)
    system_message = f"""你是一个乐于助人的知识库助手。
请仅根据以下【知识库】内容回答用户的问题。
如果知识库中没有答案，请直接说“根据现有知识无法回答”。

【知识库】
{context}
"""

    # 这里模型名改成 Ollama 里的模型名
    stream = client.chat.completions.create(
        model="llama3.2:1b",  # 👈 修改这里
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_input}
        ],
        stream=True,
        temperature=0
    )

    print("\n助手：", end="", flush=True)
    for chunk in stream:
        # Ollama 返回的字段结构和 OpenAI 一致
        content = chunk.choices[0].delta.content
        if content:
            print(content, end="", flush=True)
    print("\n")

if __name__ == "__main__":
    chat_with_knowledge()