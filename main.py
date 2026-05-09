from openai import OpenAI
import numpy as np
import pickle
import os
from dotenv import load_dotenv

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
CHAT_MODEL = "deepseek-v4-flash"
# VECTOR_CACHE_FILE = "knowledge_vectors_deepseek.pkl"

ollama_client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama"
)

EMBEDDING_MODEL = "qwen3-embedding:4b"
# CHAT_MODEL = "qwen2.5:3b" # 建议升级到3B，效果会好很多
VECTOR_CACHE_FILE = "knowledge_vectors.pkl" # 向量缓存文件名

# ================= 1. 核心工具函数 =================
def get_embedding(text):
    """调用 Ollama 获取文本的向量表示"""
    response = ollama_client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=text
    )
    return np.array(response.data[0].embedding)

def cosine_similarity(vec1, vec2):
    """计算两个向量的余弦相似度"""
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    return dot_product / (norm1 * norm2)

# ================= 2. 智能加载知识库（带缓存） =================
def load_knowledge_base(file_path):
    """读取txt文件，每一行作为一条知识存入列表"""
    with open(file_path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]

def load_or_create_vectors(knowledge_texts, cache_file):
    """
    智能加载向量：
    1. 如果缓存文件存在，直接加载
    2. 如果不存在，生成向量并保存到缓存
    """
    if os.path.exists(cache_file):
        print(f"发现向量缓存文件 {cache_file}，正在加载...")
        with open(cache_file, "rb") as f:
            cached_data = pickle.load(f)
            
        # 检查缓存是否和当前知识库一致
        if cached_data["knowledge_texts"] == knowledge_texts:
            print("缓存匹配成功！")
            return cached_data["knowledge_embeddings"]
        else:
            print("知识库已更新，需要重新生成向量...")
    
    # 缓存不存在或不匹配，重新生成
    print(f"正在使用 {EMBEDDING_MODEL} 生成知识库向量，请稍候...")
    knowledge_embeddings = [get_embedding(f"search_document: {text}") for text in knowledge_texts]
    
    # 保存到缓存
    with open(cache_file, "wb") as f:
        pickle.dump({
            "knowledge_texts": knowledge_texts,
            "knowledge_embeddings": knowledge_embeddings
        }, f)
    print(f"向量已保存到缓存文件 {cache_file}！")
    
    return knowledge_embeddings

# 程序启动时智能加载
knowledge_texts = load_knowledge_base("my_knowledge.txt")
knowledge_embeddings = load_or_create_vectors(knowledge_texts, VECTOR_CACHE_FILE)
print("知识库加载完成！\n")

# ================= 3. 检索器（带Nomic官方前缀优化） =================
def find_relevant_info(user_question, top_k=3):
    query_embedding = get_embedding(f"search_query: {user_question}")
    similarities = [cosine_similarity(query_embedding, emb) for emb in knowledge_embeddings]
    top_indices = np.argsort(similarities)[::-1][:top_k]
    results = [knowledge_texts[i] for i in top_indices]
    return "\n".join(results)

# ================= 4. 核心问答逻辑 =================
def chat_with_knowledge():
    while True:
        user_input = input("请输入你的问题（输入'退出'结束）：")
        
        if user_input.lower() in ["退出", "exit", "quit"]:
            print("再见！")
            break
        
        print("\n" + "="*50)
        print(f"[DEBUG] 用户问题：{user_input}")
        
        context = find_relevant_info(user_input, top_k=3)
        print(f"\n[DEBUG] 检索到的上下文：")
        print("-"*30)
        print(context)
        print("-"*30)
        
        final_prompt = f"""以下是参考知识库：
{context}

请严格根据上面的参考知识库回答用户的问题：{user_input}
如果知识库中没有明确提到答案，请直接说"根据现有知识无法回答"。
不要编造任何信息，不要添加任何知识库中没有的内容。"""
        
        print(f"\n[DEBUG] 发给模型的完整Prompt：")
        print("-"*30)
        print(final_prompt)
        print("-"*30)
        print("="*50 + "\n")

        stream = client.chat.completions.create(
            model=CHAT_MODEL,
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
        user_input = input("请输入你的问题（输入'退出'结束）：")
        
        if user_input.lower() in ["退出", "exit", "quit"]:
            print("再见！")
            break
        
        context = find_relevant_info(user_input, top_k=3)
        
        final_prompt = f"""以下是参考知识库：
{context}

请严格根据上面的参考知识库回答用户的问题：{user_input}
如果知识库中没有明确提到答案，请直接说"根据现有知识无法回答"。
不要编造任何信息，不要添加任何知识库中没有的内容。"""

        stream = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[{"role": "user", "content": final_prompt}],
            stream=True,
            temperature=0
        )

        print("\n助手：", end="", flush=True)
        for chunk in stream:
            content = chunk.choices[0].delta.content
            if content:
                print(content, end="", flush=True)
        print("\n")

if __name__ == "__main__":
    chat_with_knowledge()