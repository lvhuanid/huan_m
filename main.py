from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver
from typing import TypedDict, Annotated
import operator

class State(TypedDict):
    messages: Annotated[list, operator.add]
    counter: int
    user_approval: str  # 新增：用于存放人工指令（'yes' 或 'no'）

def chat_node(state: State):
    # 根据人工反馈决定逻辑
    if state.get("user_approval") == "yes":
        new_count = state.get("counter", 0) + 1
        msg = f"人工已批准，进入第 {new_count} 轮"
    else:
        new_count = state.get("counter", 0)
        msg = f"人工拒绝或未操作，保持在第 {new_count} 轮"
        
    return {
        "messages": [msg],
        "counter": new_count,
        "user_approval": None # 重置审批状态
    }

builder = StateGraph(State)
builder.add_node("chat", chat_node)
builder.set_entry_point("chat")
builder.add_edge("chat", "chat")

with SqliteSaver.from_conn_string("human_loop.db") as checkpointer:
    # 关键点：在执行 chat 节点前中断，等待人工输入
    graph = builder.compile(checkpointer=checkpointer, interrupt_before=["chat"])
    config = {"configurable": {"thread_id": "human-1"}}

    # --- 第一阶段：初始化并运行到断点 ---
    print("--- 启动工作流 ---")
    graph.invoke({"messages": ["系统启动"], "counter": 0, "user_approval": ""}, config)
    
    # 此时图会停在 chat 节点之前，不会打印任何输出，因为节点还没跑
    snapshot = graph.get_state(config)
    print(f"当前状态：{snapshot.values['messages'][-1]}，等待节点：{snapshot.next}")

    # --- 第二阶段：人工干预 (Human Input) ---
    print("\n--- 人工干预：输入审批结果 ---")
    # 模拟外部输入：手动更新线程状态
    graph.update_state(config, {"user_approval": "yes"}, as_node="chat")
    
    # --- 第三阶段：恢复运行 ---
    print("--- 恢复工作流 ---")
    result = graph.invoke(None, config)
    print(f"执行结果：{result['messages'][-1]}")
    print(f"计数器数值：{result['counter']}")