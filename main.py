from typing import TypedDict, Annotated, List, Union
import operator
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver

# 1. 定义状态
class State(TypedDict):
    # 使用 operator.add 让建议累加，而不是覆盖
    task: str
    translation: str
    review_history: Annotated[List[str], operator.add]
    score: int
    attempts: int

# 2. 节点逻辑
def translator(state: State):
    attempts = state.get("attempts", 0) + 1
    # 模拟翻译逻辑（实际开发中这里对接 LLM）
    translation = "Hello World" if attempts == 1 else "Hello to the World (Corrected)"
    print(f"--- 节点：翻译员 (第 {attempts} 次尝试) ---")
    return {
        "translation": translation,
        "attempts": attempts
    }

def reviewer(state: State):
    print(f"--- 节点：审核员 ---")
    # 模拟审核逻辑：第一次给低分，第二次给高分
    score = 7 if state["attempts"] == 1 else 9
    suggestion = f"第 {state['attempts']} 次翻译的语法可以更地道一些。"
    return {
        "score": score,
        "review_history": [suggestion]
    }

# 3. 条件边逻辑
def decide_to_finish(state: State):
    # 达到 3 次强制人工介入（由外部 interrupt 配合）
    if state["attempts"] >= 3:
        print("--- 尝试次数过多，进入人工审批状态 ---")
        return "human_break"
    # 分数达标则结束
    if state["score"] > 8:
        print("--- 审核通过 ---")
        return "finish"
    # 否则重试
    print("--- 审核未通过，准备重译 ---")
    return "retry"

# 4. 构建图
builder = StateGraph(State)

builder.add_node("translator", translator)
builder.add_node("reviewer", reviewer)

builder.set_entry_point("translator")

builder.add_edge("translator", "reviewer")

# 核心：根据审核结果跳转
builder.add_conditional_edges(
    "reviewer",
    decide_to_finish,
    {
        "finish": END,
        "retry": "translator",
        "human_break": "translator" # 即使指向下一跳，也会因为 interrupt 停在执行前
    }
)

# --- 5. 编译与运行 ---
# 使用全新的持久化方式
with SqliteSaver.from_conn_string(":memory:") as checkpointer:
    # 注意：在 compile 时传入 checkpointer
    graph = builder.compile(
        checkpointer=checkpointer, 
        interrupt_before=["translator"]
    )

    config = {"configurable": {"thread_id": "trans-1"}}

    print("开始任务...")
    # 第一次运行：初始化
    graph.invoke({"task": "你好，世界", "attempts": 0, "score": 0, "review_history": []}, config)

    print("\n[第一次唤醒：开始翻译]")
    graph.invoke(None, config)

    # 获取当前状态快照
    snapshot = graph.get_state(config)
    print(f"\n当前分数: {snapshot.values.get('score')}")
    print(f"历史建议: {snapshot.values.get('review_history')}")

    print("\n[第二次唤醒：纠错翻译]")
    result = graph.invoke(None, config)
    print(f"\n最终结果: {result['translation']}")