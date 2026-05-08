# step2_embed_store.py (升级版)
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from step1_chunk import load_and_chunk

def build_vectorstore(chunks, persist_dir: str = "chroma_db"):
    embeddings = OllamaEmbeddings(
        model="nomic-embed-text",
        base_url="http://localhost:11434"
    )
    
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_dir,
        collection_name="my_notes"
    )
    return vectorstore

if __name__ == "__main__":
    chunks = load_and_chunk("data/notes.md")
    vs = build_vectorstore(chunks)
    print(f"向量库已构建，集合内文档数: {vs._collection.count()}")