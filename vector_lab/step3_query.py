# step3_query.py (升级版)
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

def query_notes(query: str, persist_dir: str = "chroma_db", k: int = 3):
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    vectorstore = Chroma(
        persist_directory=persist_dir,
        embedding_function=embeddings,
        collection_name="my_notes"
    )
    results = vectorstore.similarity_search_with_score(query, k=k)
    return results

if __name__ == "__main__":
    q = "什么是自注意力机制？"
    for doc, score in query_notes(q):
        print(f"[相似度: {score:.4f}] {doc.page_content[:200]}...\n")