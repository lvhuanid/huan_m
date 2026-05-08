from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama"
)

def load_knowledge_base(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]

knowledge_base = load_knowledge_base("my_knowledge.txt")

def find_relevant_info(user_question):
    return "\n".join(knowledge_base)

def chat_with_knowledge():
    user_input = input("请输入你的问题：")
    
    context = find_relevant_info(user_input)
    
    # 🚩 修改点1：不再使用 system message，而是把所有内容拼在 user 里
    # 并且稍微调整了一下语气，更适合小模型
    final_prompt = f"""以下是参考知识库：
{context}

请根据上面的参考知识库，回答用户的问题：{user_input}
如果知识库里没有相关内容，请回答“根据现有知识无法回答”。"""

    # 🚩 修改点2：加入 Debug，看看发给模型的到底是什么
    print("\n[DEBUG] 发给模型的完整内容：")
    print("-"*30)
    print(final_prompt)
    print("-"*30, "\n")

    stream = client.chat.completions.create(
        model="llama3.2:1b",
        # 这里只有 user，没有 system
        messages=[{"role": "user", "content": final_prompt}],
        stream=True,
        temperature=0
    )

    print("助手：", end="", flush=True)
    for chunk in stream:
        content = chunk.choices[0].delta.content
        if content:
            print(content, end="", flush=True)
    print("\n")

if __name__ == "__main__":
    chat_with_knowledge()