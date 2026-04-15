from typing import TypedDict, Annotated, List
import operator
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_ollama import ChatOllama # 关键引入
import os
os.environ["NO_PROXY"] = "localhost,127.0.0.1"

# 1. 初始化本地模型
# base_url 默认为 http://localhost:11434
llm = ChatOllama(model="qwen3.5:cloud", temperature=0)

class State(TypedDict):
    task: str
    code: str
    security_report: str
    is_safe: bool
    iterations: Annotated[int, operator.add]

# 2. 程序员节点：真正调用 AI 写代码
def coder_node(state: State):
    print(f"\n--- [Coder] 正在编写代码... (迭代: {state['iterations'] + 1}) ---")
    
    prompt = f"针对任务: {state['task']}。编写 Python 代码。"
    if state["security_report"]:
        prompt += f"\n上一次审计失败，原因: {state['security_report']}。请修复并只返回代码。"
    
    # 调用本地 Ollama
    response = llm.invoke(prompt)
    return {"code": response.content, "iterations": 1}

# 3. 安全员节点：真正调用 AI 进行审计
def security_node(state: State):
    print("--- [Security] 正在审计代码安全... ---")
    
    prompt = f"请审计以下代码是否含有危险操作（如删除系统文件、后门等）: \n{state['code']}\n"
    prompt += "如果安全，请回复 'SAFE'。如果不安全，请说明原因。"
    
    response = llm.invoke(prompt)
    content = response.content.upper()
    
    if "SAFE" in content and len(content) < 10:
        return {"security_report": "审计通过", "is_safe": True}
    else:
        return {"security_report": response.content, "is_safe": False}

# 4. 路由逻辑 (保持不变)
def router(state: State):
    if state["is_safe"]: return "approved"
    if state["iterations"] >= 3: return "too_many_retries"
    return "rejected"

# 5. 构建图
builder = StateGraph(State)
builder.add_node("coder", coder_node)
builder.add_node("security", security_node)
builder.set_entry_point("coder")
builder.add_edge("coder", "security")
builder.add_conditional_edges("security", router, {"approved": END, "rejected": "coder", "too_many_retries": END})

# 6. 运行
with SqliteSaver.from_conn_string(":memory:") as checkpointer:
    graph = builder.compile(checkpointer=checkpointer)
    config = {"configurable": {"thread_id": "ollama-robust-1"}}
    
    # 关键：提供完整的初始状态，避免节点内部读取不存在的 Key
    initial_input = {
        "task": "写一个读取文件并删除它的 Python 脚本",
        "code": "",                # 必须初始化
        "security_report": "",      # 必须初始化
        "is_safe": False,           # 必须初始化
        "iterations": 0             # 必须初始化
    }
    
    try:
        # 使用流式或直接调用
        result = graph.invoke(initial_input, config)
        
        print("\n" + "="*30)
        print(f"最终结果: {'安全' if result['is_safe'] else '被拦截'}")
        print(f"迭代次数: {result['iterations']}")
        print(f"最终代码:\n{result['code']}")
        
    except Exception as e:
        print(f"\n运行中出错: {e}")
        # 技巧：如果出错，查看最后一次成功保存的状态
        last_state = graph.get_state(config)
        print(f"最后一次快照数据: {last_state.values}")


# --- [Coder] 正在编写代码... (迭代: 1) ---
# --- [Security] 正在审计代码安全... ---

# --- [Coder] 正在编写代码... (迭代: 2) ---
# --- [Security] 正在审计代码安全... ---

# --- [Coder] 正在编写代码... (迭代: 3) ---
# --- [Security] 正在审计代码安全... ---

# ==============================
# 最终结果: 安全
# 迭代次数: 3
# 最终代码:
# ```python
# #!/usr/bin/env python3
# import os
# import sys
# from pathlib import Path

# # 配置安全根目录
# SAFE_ROOT_NAME = "./safe_zone"
# SAFE_ROOT_DIR = Path(SAFE_ROOT_NAME)

# def init_safe_root():
#     """初始化安全根目录，并进行严格的安全检查"""
#     # 1. 如果目录不存在，创建它，并设置严格权限 (仅所有者可读写执行)
#     if not SAFE_ROOT_DIR.exists():
#         try:
#             SAFE_ROOT_DIR.mkdir(mode=0o700, exist_ok=True)
#             # 显式设置权限以防 umask 影响
#             os.chmod(SAFE_ROOT_DIR, 0o700)
#         except OSError as e:
#             print(f"Error creating safe root directory: {e}")
#             sys.exit(1)
    
#     # 2. 关键安全检查：根目录本身不能是符号链接
#     # 防止攻击者将 safe_zone  symlink 到 / 或其他系统目录
#     if SAFE_ROOT_DIR.is_symlink():
#         print("Security Alert: Safe root directory is a symlink! Aborting.")
#         sys.exit(1)
    
#     # 3. 确保它是一个目录
#     if not SAFE_ROOT_DIR.is_dir():
#         print("Security Alert: Safe root path is not a directory! Aborting.")
#         sys.exit(1)
    
#     # 4. 获取根目录的真实绝对路径 (用于后续 containment 检查)
#     # 此时已确认 SAFE_ROOT_DIR 不是 symlink，resolve 是安全的
#     return SAFE_ROOT_DIR.resolve()

# def get_secure_file_path(root_path, filename):
#     """构建并验证文件的安全路径"""
#     # 1. 防止路径遍历攻击：只接受文件名，丢弃任何目录部分
#     safe_filename = os.path.basename(filename)
#     if not safe_filename:
#         raise ValueError("Invalid filename")
    
#     target_path = root_path / safe_filename
    
#     # 2. 检查文件是否存在
#     if not target_path.exists():
#         raise FileNotFoundError(f"File '{safe_filename}' not found in safe zone.")
    
#     # 3. 关键安全检查：目标文件不能是符号链接
#     # 防止攻击者通过 symlink 指向系统文件 (如 /etc/passwd)
#     if target_path.is_symlink():
#         raise ValueError("Security Alert: Target file is a symlink! Aborting.")
    
#     # 4. 路径 containment 检查
#     # 确保解析后的真实路径仍在安全根目录内
#     # 这可以防止通过目录内的 symlink 跳转出根目录的情况
#     real_root = str(root_path)
#     real_target = str(target_path.resolve())
    
#     # 确保目标路径以根目录路径开头，且后有分隔符 (防止 /safe_zone_evil 匹配 /safe_zone)
#     if not (real_target.startswith(real_root + os.sep) or real_target == real_root):
#         raise ValueError("Security Alert: Path traversal detected! Aborting.")
    
#     return target_path

# def read_file_secure(path):
#     """安全地读取文件内容"""
#     try:
#         # 使用 os.open 配合 O_NOFOLLOW 标志，确保打开过程中不跟随符号链接
#         flags = os.O_RDONLY
#         if hasattr(os, 'O_NOFOLLOW'):
#             flags |= os.O_NOFOLLOW
#         else:
#             # 如果系统不支持 O_NOFOLLOW (如某些 Windows 版本)，依赖之前的 is_symlink 检查
#             pass
        
#         fd = os.open(str(path), flags)
#         with os.fdopen(fd, 'r', encoding='utf-8', errors='replace') as f:
#             return f.read()
#     except OSError as e:
#         # 捕获 O_NOFOLLOW 导致的错误 (ELOOP)
#         if "Too many levels of symbolic links" in str(e) or e.errno == 40:
#             raise ValueError("Security Alert: Symlink detected during open!")
#         raise e

# def delete_file_secure(path):
#     """安全地删除文件"""
#     # 1. TOCTOU 缓解：在执行删除前再次检查是否为符号链接
#     if path.is_symlink():
#         raise ValueError("Security Alert: File became a symlink before deletion! Aborting.")
    
#     # 2. 执行删除
#     path.unlink()

# def main():
#     if len(sys.argv) < 2:
#         print(f"Usage: {sys.argv[0]} <filename>")
#         sys.exit(1)
    
#     user_filename = sys.argv[1]
    
#     # 初始化并验证安全根目录
#     secure_root = init_safe_root()
    
#     try:
#         # 构建并验证安全路径
#         target_file = get_secure_file_path(secure_root, user_filename)
        
#         # 读取文件内容
#         print(f"Reading file: {target_file.name}")
#         content = read_file_secure(target_file)
#         print("--- File Content ---")
#         print(content)
#         print("--- End Content ---")
        
#         # 二次确认机制
#         confirm = input(f"\nAre you sure you want to DELETE '{target_file.name}'? (yes/no): ")
#         if confirm.lower() not in ['yes', 'y']:
#             print("Operation cancelled by user.")
#             sys.exit(0)
        
#         # 执行删除
#         delete_file_secure(target_file)
#         print("File deleted successfully.")
        
#     except (ValueError, FileNotFoundError, OSError) as e:
#         print(f"Error: {e}")
#         sys.exit(1)
#     except KeyboardInterrupt:
#         print("\nOperation interrupted.")
#         sys.exit(1)

# if __name__ == "__main__":
#     main()
# ```