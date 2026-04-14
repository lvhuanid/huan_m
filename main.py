from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_community.vectorstores import FAISS
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate

import os
os.environ["NO_PROXY"] = "localhost,127.0.0.1"

# =================配置区域=================
# 1. 设置你的文档路径（请确保该文件存在）
# 如果没有PDF，可以放一个 test.txt 试试
FILE_PATH = "员工手册.pdf" 
# 如果上面是txt文件，记得把 PyPDFLoader 换成 TextLoader

# 2. 初始化模型
# 嵌入模型：负责把文字变成向量（推荐 nomic-embed-text，如果没有需先 ollama pull nomic-embed-text）
# 如果不想下载新模型，也可以用 qwen2.5 或默认的，但效果可能稍差
embeddings = OllamaEmbeddings(model="nomic-embed-text") 

# 对话模型：负责回答问题
llm = ChatOllama(model="qwen3.5:cloud", temperature=0.7)

# =================核心流程=================

def process_document(file_path):
    """
    步骤 1 & 2: 加载文档并切分
    """
    print(f"📂 正在加载文档: {file_path}...")
    
    # 判断文件类型加载
    if file_path.endswith(".pdf"):
        loader = PyPDFLoader(file_path)
    else:
        loader = TextLoader(file_path, encoding="utf-8")
    
    documents = loader.load()
    
    # 切分文本
    # chunk_size=500: 每块500个字符（大约200-300汉字）
    # chunk_overlap=50: 块与块之间重叠50个字符，防止切断语义
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_documents(documents)
    
    print(f"✅ 文档已切分为 {len(chunks)} 个片段")
    return chunks

def create_vector_store(chunks):
    """
    步骤 3: 向量化并存入 FAISS
    """
    print("🧠 正在向量化并构建索引（这可能需要一点时间）...")
    
    # 从文本块创建向量存储
    # 这一步会调用 Ollama 把每一段文字变成向量
    vectorstore = FAISS.from_documents(chunks, embeddings)
    
    # 保存到本地（可选，下次就不用重新算向量了）
    vectorstore.save_local("faiss_index")
    print("✅ 向量库构建完成！")
    return vectorstore

def build_rag_chain(vectorstore):
    """
    步骤 4: 构建问答链条
    """
    # 设置检索器：每次找最相关的 3 个片段
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    
    # 定义提示词模板（关键！）
    # 告诉 AI：你要根据我给你的 context 来回答，不要瞎编
    template = """
    你是一个专业的文档分析助手。请根据下方的【参考信息】来回答【问题】。
    如果【参考信息】中没有答案，请直接说“文档中未找到相关信息”，不要编造。
    
    【参考信息】：
    {context}
    
    【问题】：
    {question}
    
    【回答】：
    """
    prompt = ChatPromptTemplate.from_template(template)
    
    # 构建处理链条
    # 流程：用户问题 -> 检索器(找资料) -> 填充模板 -> 大模型(生成回答)
    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
    )
    return rag_chain

# =================主程序入口=================
if __name__ == "__main__":
    # 1. 检查文件是否存在
    if not os.path.exists(FILE_PATH):
        print(f"❌ 错误：找不到文件 {FILE_PATH}，请先创建一个测试文件！")
    else:
        # 2. 处理文档
        chunks = process_document(FILE_PATH)
        
        # 3. 建库
        vectorstore = create_vector_store(chunks)
        
        # 4. 建链
        chain = build_rag_chain(vectorstore)
        
        # 5. 开始对话循环
        print("\n🤖 机器人：文档处理完毕，请问有什么可以帮您？（输入 'quit' 退出）")
        while True:
            query = input("\n👤 用户：")
            if query.lower() == 'quit':
                break
            
            # 流式输出体验更好（可选）
            print("🤖 机器人：", end="", flush=True)
            for chunk in chain.stream(query):
                print(chunk.content, end="", flush=True)
            print() # 换行