from langchain_ollama import OllamaEmbeddings

embeddings = OllamaEmbeddings(model="nomic-embed-text")
vector = embeddings.embed_query("AI Ops 是未来")
print(f"向量长度: {len(vector)}") # nomic 通常是 768 维
print(f"前 5 位数值: {vector[:5]}")

# 知识点：理解向量空间、余弦相似度（Cosine Similarity）。

# 核心逻辑：AI 不认识文字，它只认识坐标。Embedding 模型将一句话映射到高维空间的坐标。语义相近的话，坐标距离就近。