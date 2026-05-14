import os
from openai import OpenAI
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
import chromadb
from chromadb.config import Settings




load_dotenv()

MODEL = os.getenv("LLM_MODEL_ID")
client = OpenAI(
    api_key=os.getenv("MODELSCOPE_API_KEY"),
    base_url=os.getenv("LLM_BASE_URL")
)
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")
ollama_client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama"
)
#   """调用 Ollama 获取文本的向量表示"""
# response = ollama_client.embeddings.create(
#     model=EMBEDDING_MODEL,
#     input=text
# )
# resp =  ollama_client.embeddings.create(
#     model=EMBEDDING_MODEL,
#     input=["今天天气真好", "下雨了，好冷"]
# )

# Chroma 会持久化到本地目录
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(name="resumes")

def embed_texts(texts):
    resp = ollama_client.embeddings.create(model=EMBEDDING_MODEL, input=texts)
    return [d.embedding for d in resp.data]

# -------------------- 数据加载与入库 --------------------
# 读取文件
# with open("handbook.txt", "r", encoding="utf-8") as f:
#     full_text = f.read()

# 分块
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100,
    separators=["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""]
)
# chunks = text_splitter.split_text(full_text)

# 向量库
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(name="handbook")

# 入库
# batch_size = 20
# for i in range(0, len(chunks), batch_size):
#     batch = chunks[i:i+batch_size]
#     embs = embed_texts(batch)
#     collection.upsert(
#         documents=batch,
#         embeddings=embs,
#         ids=[f"chunk_{i+j}" for j in range(len(batch))]
#     )
# print(f"已存入 {len(chunks)} 个文本块")

# -------------------- 检索与问答 --------------------
def retrieve(query, n_results=4):
    q_emb = embed_texts([query])[0]
    res = collection.query(query_embeddings=[q_emb], n_results=n_results)
    return res['documents'][0] if res['documents'] else []

def ask_handbook(question):
    docs = retrieve(question)
    context = "\n\n---\n\n".join(docs)
    prompt = f"""你是公司的员工手册问答助手。请根据以下手册内容回答用户的问题。
如果内容中没有答案，请明确告知“手册中未找到相关信息”，不要编造任何条款。

手册内容：
{context}

问题：{question}
答案："""

    resp = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        stream=True
    )
    print("AI: ", end='', flush=True)
    full = ""
    for chunk in resp:
        if chunk.choices[0].delta.content:
            print(chunk.choices[0].delta.content, end='', flush=True)
            full += chunk.choices[0].delta.content
    print()
    return full

# -------------------- 交互 --------------------
if __name__ == "__main__":
    print("员工手册问答（输入 exit 退出）")
    while True:
        q = input("问: ")
        if q.lower() in ("exit", "quit"):
            break
        ask_handbook(q)