import os
from typing import List
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_ollama import OllamaLLM
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever

from unstructured.partition.pdf import partition_pdf
from langchain_core.documents import Document


os.environ["NO_PROXY"] = "localhost,127.0.0.1"


# --- 1. 自定义混合检索器类 ---
# 继承 BaseRetriever 让我们能够完全控制检索逻辑
class CustomHybridRetriever(BaseRetriever):
    vector_retriever: BaseRetriever
    bm25_retriever: BaseRetriever
    top_k: int = 3

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        # 分别获取向量检索和关键词检索的结果
        vector_docs = self.vector_retriever.invoke(query)
        bm25_docs = self.bm25_retriever.invoke(query)

        # 简单的合并与去重逻辑
        combined_docs = vector_docs + bm25_docs
        
        # 根据内容去重
        unique_docs = []
        seen_content = set()
        for doc in combined_docs:
            if doc.page_content not in seen_content:
                unique_docs.append(doc)
                seen_content.add(doc.page_content)
        
        return unique_docs[:self.top_k]


def process_pdf_with_images(file_path: str):
    print(f"🚀 正在使用 Unstructured 解析 PDF: {file_path}")
    
    # 1. 核心解析逻辑
    elements = partition_pdf(
        filename=file_path,
        # 策略设为 hi_res 会触发布局检测
        strategy="hi_res", 
        
        # 识别图片中的文字 (OCR)
        infer_table_structure=True,
        # extract_images_in_pdf=True, 
        
        # 这里的模型通常会自动下载 (如 yolo, detectron2)
        hi_res_model_name="yolox",
        languages=["chi_sim", "eng"],

        # 图片保存路径（如果需要单独查看提取出的图片）
        # extract_image_block_output_dir="extracted_images",
    )

    # 2. 将解析出的 Element 转换为 LangChain 的 Document
    docs = []
    for element in elements:
        # element 类型可能是 Table, NarrativeText, Image, Title 等
        # Unstructured 已经帮我们将图片里的文字通过 OCR 转成了文本
        metadata = element.metadata.to_dict()
        metadata["type"] = element.category
        
        new_doc = Document(
            page_content=element.text,
            metadata=metadata
        )
        docs.append(new_doc)
    
    return docs
# --- 2. 环境准备与模型初始化 ---
# 确保你的 Ollama 已启动，并且有这两个模型
embeddings = OllamaEmbeddings(model="nomic-embed-text")
llm = OllamaLLM(model="qwen3.5:cloud", temperature=0)

# --- 3. 数据处理 (实践步骤：数据清洗) ---
raw_documents = [
    "项目代码：PRJ-2026。负责人：张三。状态：进行中。",
    "公司地址：北京市朝阳区科技园 A 座。联系电话：010-123456.",
    "产品特性：支持 8K 视频录制，配备自研 M3 芯片，续航 20 小时。",
    "客服部小李负责售后咨询，工作时间 9:00 - 18:00。"
]

# 1. 使用 Unstructured 解析 PDF (包含图片 OCR 逻辑)
pdf_docs = process_pdf_with_images("su7.pdf")

# 2. 初始化切分器
text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50)

# 3. 关键修改：对解析出的 PDF 文档进行切分
# 注意：这里使用 split_documents 而不是 create_documents
chunks = text_splitter.split_documents(pdf_docs)

print(f"✅ PDF 解析完成，切分为 {len(chunks)} 个数据块。")


# --- 4. 构建检索组件 ---
# 创建向量库
vectorstore = FAISS.from_documents(chunks, embeddings)
v_retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# 创建 BM25 检索器（针对关键词）
b_retriever = BM25Retriever.from_documents(chunks)
b_retriever.k = 3

# 实例化我们自定义的检索器
my_hybrid_retriever = CustomHybridRetriever(
    vector_retriever=v_retriever,
    bm25_retriever=b_retriever,
    top_k=3
)

# --- 5. RAG 执行流程 ---
def run_rag_pipeline(question: str):
    print(f"\n🔍 正在检索: {question}")
    
    # 获取检索到的文档
    retrieved_docs = my_hybrid_retriever.invoke(question)
    
    # 构造上下文
    context = "\n---\n".join([doc.page_content for doc in retrieved_docs])
    
    # 构造 Prompt
    prompt = f"""你是一个专业的助手。请根据以下参考资料回答问题。
资料内容：
{context}

问题：{question}
请基于资料给出简洁、准确的回答。如果资料中没提到，请直说不知道。"""

    print("🤖 模型正在思考...")
    response = llm.invoke(prompt)
    return response

# --- 6. 测试运行 ---
if __name__ == "__main__":
    query = "外观颜色有哪些？能源类型是什么？"
    result = run_rag_pipeline(query)
    print("\n✨ 最终回答:")
    print(result)