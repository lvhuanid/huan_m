from openai import OpenAI
import numpy as np

# ================= 配置区域 =================
client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama"
)
EMBEDDING_MODEL = "nomic-embed-text:latest" # 👈 已修改
CHAT_MODEL = "qwen2.5:3b"

# ================= 1. 核心工具函数 =================
def get_embedding(text):
    """调用 Ollama 获取文本的向量表示"""
    response = client.embeddings.create(
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

# ================= 2. 数据加载与向量化 =================
def load_knowledge_base(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]

knowledge_texts = load_knowledge_base("my_knowledge.txt")
print(f"正在使用 {EMBEDDING_MODEL} 生成知识库向量，请稍候...")
knowledge_embeddings = [get_embedding(text) for text in knowledge_texts]
print("向量生成完毕！\n")

# ================= 3. 真正的检索器 =================
# ================= 3. 真正的检索器（带 Debug） =================
def find_relevant_info(user_question, top_k=1):
    query_embedding = get_embedding(user_question)
    
    # 计算相似度列表
    similarities = [cosine_similarity(query_embedding, emb) for emb in knowledge_embeddings]
    
    # 🚩 Debug 开始：打印所有知识及其相似度
    print("\n[DEBUG] 知识库相似度排名：")
    print("-" * 50)
    # 把文本、相似度、索引打包在一起排序
    scored_texts = sorted(
        zip(knowledge_texts, similarities, range(len(knowledge_texts))),
        key=lambda x: x[1],
        reverse=True
    )
    for text, score, idx in scored_texts:
        print(f"[相似度: {score:.4f}] {text}")
    print("-" * 50, "\n")
    # 🚩 Debug 结束
    
    # 找到相似度最高的索引
    top_indices = np.argsort(similarities)[::-1][:top_k]
    
    # 取出对应的文本
    results = [knowledge_texts[i] for i in top_indices]
    return "\n".join(results)

# ================= 4. 核心问答逻辑 =================
def chat_with_knowledge():
    user_input = input("请输入你的问题：")
    context = find_relevant_info(user_input, top_k=3)
    
    final_prompt = f"""以下是参考知识库：
{context}

请根据上面的参考知识库，回答用户的问题：{user_input}
如果知识库里没有相关内容，请回答“根据现有知识无法回答”。"""

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