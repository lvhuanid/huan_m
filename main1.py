import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
    api_key=os.getenv("MODELSCOPE_API_KEY"),
    base_url=os.getenv("LLM_BASE_URL")
)
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")
ollama_client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama"
)
#   """调用 Ollama 获取文本的向量表示"""
# response = ollama_client.embeddings.create(
#     model=EMBEDDING_MODEL,
#     input=text
# )
resp =  ollama_client.embeddings.create(
    model=EMBEDDING_MODEL,
    input=["今天天气真好", "下雨了，好冷"]
)
vec1 = resp.data[0].embedding  # 一个长列表，比如 1536 维
vec2 = resp.data[1].embedding

model = os.getenv("LLM_MODEL_ID")

# response = client.chat.completions.create(
#     model=model,
#     messages=[
#         {"role": "system", "content": "你是一个乐于助人的助手"},
#         {"role": "user", "content": "你好，请用一句话介绍你自己"}
#     ],
#     temperature=0.7,
#     max_tokens=100
# )

# # temperature=0 → 输出严谨、逻辑化
# # temperature=1.5 → 更发散，可能会有比喻或脑洞
# # max_tokens=50 → 话说一半就被截断

# print(response.choices[0].message.content)

# messages = [{"role": "system", "content": "你是儿童科普老师，回答要简单有趣"}]
# while True:
#     user_input = input("你: ")
#     messages.append({"role": "user", "content": user_input})
#     stream = client.chat.completions.create(
#         model=model,
#         messages=messages,
#         temperature=0.8,
#         stream=True
#     )
#     for chunk in stream:
#         delta = chunk.choices[0].delta
#         if delta.content:
#             print(delta.content, end='', flush=True)

    # Traceback (most recent call last):
    # File "/Users/huan/dev/llm/main1.py", line 51, in <module>
    #     delta = chunk.choices[0].delta
    #             ~~~~~~~~~~~~~^^^
    # IndexError: list index out of range

    # print(f"AI: {assistant_reply}")
    # messages.append({"role": "assistant", "content": assistant_reply})

# 预设角色及其 system prompt
ROLES = {
    "1": {
        "name": "心理咨询师",
        "system_prompt": "你是一位温暖、专业的心理咨询师。请以共情的方式回应用户，适当提问，帮助ta梳理情绪。"
    },
    "2": {
        "name": "段子手",
        "system_prompt": "你是一个油麦的脱口秀演员，擅长用幽默的方式吐槽生活中的小事。回答要风趣，可以带梗。"
    },
    "3": {
        "name": "面试官",
        "system_prompt": "你是一位严格的互联网大厂面试官，主要考察候选人的系统设计能力。请提出有挑战性的问题，并对回答进行追问。"
    }
}

def choose_role():
    print("请选择聊天角色：")
    for key, role in ROLES.items():
        print(f"{key}. {role['name']}")
    while True:
        choice = input("输入编号（1/2/3）：").strip()
        if choice in ROLES:
            return ROLES[choice]
        print("无效选择，请重新输入。")

def chat_loop(system_prompt):
    # 初始化消息列表，只包含系统设定
    messages = [{"role": "system", "content": system_prompt}]
    print(f"\n开始聊天（输入 exit 或 quit 退出）\n")

    while True:
        try:
            user_input = input("你: ")
        except EOFError:
            break  # 处理 Ctrl+D

        if user_input.lower() in ("exit", "quit"):
            print("聊天结束，再见！")
            break

        # 将用户消息加入历史
        messages.append({"role": "user", "content": user_input})

        try:
            # 流式调用
            stream = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.8,
                stream=True
            )
        except Exception as e:
            print(f"API 调用失败: {e}")
            # 调用失败时移除刚刚加入的用户消息，避免破坏历史
            messages.pop()
            continue

        # 打印助手前缀
        print("AI: ", end='', flush=True)

        # 收集完整回复
        full_reply = ""

        for chunk in stream:
            delta = chunk.choices[0].delta
            if delta.content:
                print(delta.content, end='', flush=True)
                full_reply += delta.content

        print()  # 换行

        # 将完整回复加入历史，供下一轮使用
        messages.append({"role": "assistant", "content": full_reply})

if __name__ == "__main__":
    role = choose_role()
    chat_loop(role["system_prompt"])
