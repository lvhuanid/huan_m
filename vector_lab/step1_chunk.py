# step1_chunk.py (升级版)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

def load_and_chunk(file_path: str, chunk_size: int = 500, chunk_overlap: int = 80):
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()
    doc = Document(page_content=text, metadata={"source": file_path})
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n## ", "\n### ", "\n", " ", ""]
    )
    chunks = splitter.split_documents([doc])
    return chunks

if __name__ == "__main__":
    chunks = load_and_chunk("data/notes.md")
    print(f"切分出 {len(chunks)} 个文档块")
    for i, chunk in enumerate(chunks):
        print(f"Chunk {i}: {chunk.page_content[:100]}...\n")