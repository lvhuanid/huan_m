import chromadb
from chromadb.utils import embedding_functions

# 1. 初始化本地持久化客户端
client = chromadb.PersistentClient(path="./my_vector_db")

# 2. 创建集合 (类似数据库的表)
# 注意：这里我们使用本地 Ollama 作为向量生成引擎
collection = client.get_or_create_collection(name="tech_docs")

# 3. 模拟存入技术文档
collection.add(
    documents=["Docker 容器重启命令是 docker restart", "Kubernetes 是容器编排引擎"],
    metadatas=[{"source": "ops_manual"}, {"source": "k8s_doc"}],
    ids=["id1", "id2"]
)

# 4. 语义检索
results = collection.query(
    query_texts=["如何让容器重新运行？"],
    n_results=1
)
print(f"检索结果: {results['documents']}")