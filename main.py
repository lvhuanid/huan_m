from openai import OpenAI
import numpy as np
import pickle
import os
from dotenv import load_dotenv
from datetime import datetime
import json



load_dotenv()  # 从 .env 文件加载环境变量

# ================= 配置区域 =================
# client = OpenAI(
#     base_url="http://localhost:11434/v1",
#     api_key="ollama"
# )
client = OpenAI(
    base_url="https://api.deepseek.com",
    api_key=os.environ.get("DEEPSEEK_API_KEY")
)
# EMBEDDING_MODEL = "deepseek-embedding"
MODEL = "deepseek-v4-flash"
# VECTOR_CACHE_FILE = "knowledge_vectors_deepseek.pkl"

# ollama_client = OpenAI(
#     base_url="http://localhost:11434/v1",
#     api_key="ollama"
# )

# EMBEDDING_MODEL = "qwen3-embedding:4b"
# # CHAT_MODEL = "qwen2.5:3b" # 建议升级到3B，效果会好很多
# VECTOR_CACHE_FILE = "knowledge_vectors.pkl" # 向量缓存文件名

def get_current_time():
    """获取当前系统时间"""
    return f"当前时间是：{datetime.now().strftime('%Y年%m月%d日 %H:%M:%S')}"

def calculate_expression(expression):
    """计算数学表达式的值
    Args:
        expression: 要计算的数学表达式，如"2+3*4"
    """
    try:
        # 注意：eval有安全风险，这里仅用于演示
        result = eval(expression)
        return f"计算结果：{expression} = {result}"
    except Exception as e:
        return f"计算错误：{str(e)}"

# ================= 2. 告诉模型我们有哪些工具 =================
# 这是一个标准的OpenAI工具定义格式
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "获取当前系统时间",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calculate_expression",
            "description": "计算数学表达式的值",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "要计算的数学表达式，如'2+3*4'"
                    }
                },
                "required": ["expression"]
            }
        }
    }
]

# ================= 3. 核心工具调用逻辑 =================
def chat_with_tools():
    messages = [
        # 🔴 新增：强制模型使用工具的系统提示词
        {"role": "system", "content": """你是一个严格按照流程工作的智能助手。
你的工作流程必须是：
1. 分析用户的问题，判断是否需要调用工具
2. 如果需要，调用对应的工具获取结果
3. **最重要：根据工具返回的结果，进行必要的推理和计算，然后回答用户的完整问题**
4. 绝对不能只返回工具的原始输出，必须用自然语言整理并回答用户的完整问题

例如：
用户问："现在的时间加3小时是什么时候？"
你应该：
- 调用get_current_time工具，得到"2026年05月09日 17:14:38"
- 自己计算：17:14:38 + 3小时 = 20:14:38
- 回答："现在的时间是2026年05月09日 17:14:38，加3小时后是2026年05月09日 20:14:38"

对于任何需要获取实时信息或计算的问题，你必须调用对应的工具，绝对不能直接回答。
即使你知道答案，也必须先调用工具获取结果，再根据结果回答。"""}
    ]
    
    while True:
        user_input = input("请输入你的问题（输入'退出'结束）：")
        
        if user_input.lower() in ["退出", "exit", "quit"]:
            print("再见！")
            break
        
        messages.append({"role": "user", "content": user_input})
        
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=tools,
            tool_choice="auto",
            stream=False
        )
        
        response_message = response.choices[0].message
        messages.append(response_message)
        
        if response_message.tool_calls:
            print("\n[DEBUG] 模型要求调用工具：")
            
            for tool_call in response_message.tool_calls:
                function_name = tool_call.function.name
                function_args = json.loads(tool_call.function.arguments)
                
                print(f"  调用函数：{function_name}")
                print(f"  参数：{function_args}")
                
                if function_name == "get_current_time":
                    function_response = get_current_time()
                elif function_name == "calculate_expression":
                    function_response = calculate_expression(**function_args)
                else:
                    function_response = f"未知函数：{function_name}"
                
                print(f"  执行结果：{function_response}\n")
                
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": function_name,
                    "content": function_response
                })
            
            second_response = client.chat.completions.create(
                model=MODEL,
                messages=messages,
                stream=True
            )
            
            print("助手：", end="", flush=True)
            for chunk in second_response:
                content = chunk.choices[0].delta.content
                if content:
                    print(content, end="", flush=True)
            print("\n")
        
        else:
            print("\n助手：", end="", flush=True)
            print(response_message.content)
            print("\n")

if __name__ == "__main__":
    chat_with_tools()