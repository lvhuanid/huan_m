from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver
from typing import TypedDict, Annotated
import operator

class State(TypedDict):
    messages: Annotated[list, operator.add]
    counter: int

def chat_node(state: State):
    new_count = state.get("counter", 0) + 1
    return {
        "messages": [f"第 {new_count} 轮对话"],
        "counter": new_count
    }

builder = StateGraph(State)
builder.add_node("chat", chat_node)
builder.set_entry_point("chat")
builder.add_edge("chat", "chat") # 修改点：让节点指向自身形成循环

with SqliteSaver.from_conn_string("demo_fix.db") as checkpointer:
    # 修改点：在 chat 节点执行后设置中断，防止死循环并保存状态
    graph = builder.compile(checkpointer=checkpointer, interrupt_after=["chat"])
    
    config = {"configurable": {"thread_id": "user-456"}}
    
    print("=== 第一次对话 ===")
    result1 = graph.invoke({"messages": [], "counter": 0}, config)
    print(f"消息: {result1['messages']}")
    print(f"计数器: {result1['counter']}")
    
    print("\n=== 第二次对话 ===")
    result2 = graph.invoke(None, config)
    print(f"消息: {result2['messages']}")
    print(f"计数器: {result2['counter']}")
    
    print("\n=== 第三次对话 ===")
    result3 = graph.invoke(None, config)
    print(f"消息: {result3['messages']}")
    print(f"计数器: {result3['counter']}")