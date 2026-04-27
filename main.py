import os
from datetime import datetime
from langchain.tools import tool
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_ollama import ChatOllama



os.environ["NO_PROXY"] = "localhost,127.0.0.1"

llm = ChatOllama(model="qwen3.5:cloud", temperature=0)

@tool
def get_weekday(date: str) -> str:
    """
    根据输入的日期返回对应的星期几。
    日期格式必须为 YYYY-MM-DD，例如 2025-12-12。
    """
    try:
        dt = datetime.strptime(date, "%Y-%m-%d")
        weekdays = ["星期一", "星期二", "星期三", "星期四", "星期五", "星期六", "星期日"]
        return weekdays[dt.weekday()]
    except ValueError:
        return "请输入YYYY-MM-DD格式的日期"

tools = [get_weekday]

system_prompt = """你是一个日期查询助手。
- 如果用户询问某个日期是星期几，请调用 get_weekday 工具查询。
- 工具接收的日期参数必须是 'YYYY-MM-DD' 格式。
- 如果用户输入的日期格式不正确，请提示“请输入YYYY-MM-DD格式的日期”。
- 用中文回答用户。"""

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("user", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad")
])

agent = create_tool_calling_agent(llm, tools, prompt)

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    handle_parsing_errors=True,
    max_iterations=3
)

if __name__ == "__main__":
    response = agent_executor.invoke({"input": "2025-12-12是星期几"})
    print("回答:", response["output"])
    response = agent_executor.invoke({"input": "12/12/2025是星期几"})
    print("回答:", response["output"])