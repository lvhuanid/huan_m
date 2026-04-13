from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
import os
os.environ["NO_PROXY"] = "localhost,127.0.0.1"

# 1. 初始化模型
llm = ChatOllama(model="qwen3.5:cloud", temperature=0.7)

# 2. 定义提示词
# 注意：variable_name="chat_history" 定义了历史消息在 prompt 中的变量名
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个智能助手。"),
    MessagesPlaceholder(variable_name="chat_history"), # 坑位名字叫 chat_history
    ("human", "{input}") # 输入名字叫 input
])

# 3. 构建基础链条
chain = prompt | llm

# 4. 包装记忆模块
def get_session_history(session_id: str):
    return ChatMessageHistory(session_id=session_id)

agent_with_history = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="input",        # 告诉它：输入字典里的 "input" 是用户的话
    history_messages_key="chat_history" # 告诉它：历史消息要填进 prompt 的 "chat_history" 坑位里
)

# 5. 调用 (必须传字典！)
config = {"configurable": {"session_id": "test_session_1"}}

# 第一句
response = agent_with_history.invoke(
    {"input": "我叫阿强"}, 
    config=config
)
print(response.content)

# 第二句 (测试记忆)
response2 = agent_with_history.invoke(
    {"input": "我叫什么名字？"}, 
    config=config
)
print(response2.content)