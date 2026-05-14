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

def chunk_text(text: str, chunk_size=300, overlap=50) -> list[str]:
    """将长文本切成小块，保留重叠以保持上下文"""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", "。", "，", " ", ""]
    )
    return splitter.split_text(text)

def embed_texts(texts: list[str]) -> list[list[float]]:
    """把文本列表转成向量列表"""
    resp = ollama_client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=texts
    )
    return [d.embedding for d in resp.data]

# 模拟多份简历文本（可以改成从文件读取或调用之前的 parse_resume 生成）
resumes = [
    {
        "name": "张三",
        "skills": "React, TypeScript, Webpack",
        "experience": "腾讯科技 前端工程师 2020至今；阿里巴巴 前端实习 2018-2020"
    },
    {
        "name": "李四",
        "skills": "Python, Django, PostgreSQL, Docker",
        "experience": "字节跳动 后端工程师 2019至今"
    },
    {
        "name": "王五",
        "skills": "UI设计, Figma, Sketch, 用户体验",
        "experience": "美团 交互设计师 2021至今"
    }
]

# 为每份简历生成一段描述文本，作为检索单元
documents = []
metadatas = []
ids = []
for i, r in enumerate(resumes):
    text = f"姓名：{r['name']}\n技能：{r['skills']}\n工作经历：{r['experience']}"
    # 如果文本较长可以分块，这里简单整块存入
    chunks = chunk_text(text, chunk_size=300, overlap=50)
    for j, chunk in enumerate(chunks):
        doc_id = f"resume_{i}_{j}"
        documents.append(chunk)
        metadatas.append({"name": r["name"], "type": "resume"})
        ids.append(doc_id)

# 批量嵌入并存入 Chroma
if documents:
    embeddings = embed_texts(documents)
    collection.upsert(
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas,
        ids=ids
    )
    print(f"已存入 {len(documents)} 个文本块")
else:
    print("没有文档可存入")


def retrieve(query: str, n_results=3) -> list[str]:
    """根据问题检索最相关的文档块"""
    q_embedding = embed_texts([query])[0]
    results = collection.query(
        query_embeddings=[q_embedding],
        n_results=n_results
    )
    # results['documents'] 是一个二维列表，我们取第一个查询的结果
    return results['documents'][0] if results['documents'] else []

def ask(question: str) -> str:
    # 检索相关上下文
    retrieved_docs = retrieve(question)
    context = "\n\n---\n\n".join(retrieved_docs)

    # 构造 Prompt
    prompt = f"""你是一名招聘助手，你需要根据提供的简历资料回答用户的问题。
如果资料中没有答案，请明确说“未找到相关信息”，不要编造。

简历资料：
{context}

用户问题：{question}
答案："""

    stream = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        stream=True
    )

    print("AI: ", end='', flush=True)
    full_reply = ""
    for chunk in stream:
        if chunk.choices[0].delta.content:
            text = chunk.choices[0].delta.content
            print(text, end='', flush=True)
            full_reply += text
    print()
    return full_reply


if __name__ == "__main__":
    print("简历问答系统（输入 exit 退出）")
    while True:
        q = input("问: ")
        if q.lower() in ("exit", "quit"):
            break
        ask(q)