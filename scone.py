from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import faiss
import urllib
import numpy as np

FOLDER = 'scone_data'
KNOWLEDGE_FILE = f'{FOLDER}/scone_knowledge.txt'
INDEX_FILE = f'{FOLDER}/scone_index.faiss'
DOCS_FILE = f'{FOLDER}/scone_docs.npy'


def preprocess():
    # 讀取知識庫
    with open(KNOWLEDGE_FILE, "r", encoding="utf-8") as f:
        documents = [line.strip() for line in f if line.strip()]

    # 初始化檢索器模型
    retriever = SentenceTransformer('all-MiniLM-L6-v2')  # 小型高效模型

    # 將文本轉為向量
    doc_embeddings = retriever.encode(documents, convert_to_numpy=True)

    # 建立 FAISS 索引
    dimension = doc_embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)  # L2 距離索引
    index.add(doc_embeddings)  # 添加向量到索引

    # 保存索引（可選）
    faiss.write_index(index, INDEX_FILE)
    np.save(DOCS_FILE, documents)  # 保存原始文本

# ================

def rag_query(query, top_k=1):
    # 加載生成模型和 tokenizer
    generator_tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large")
    generator = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-large")

    # 加載檢索器和索引
    retriever = SentenceTransformer('all-MiniLM-L6-v2')
    index = faiss.read_index(INDEX_FILE)
    documents = np.load(DOCS_FILE, allow_pickle=True)

    # 將查詢轉為向量
    query_embedding = retriever.encode([query], convert_to_numpy=True)

    # 檢索相關文檔
    distances, indices = index.search(query_embedding, top_k)
    print(distances, indices)
    retrieved_docs = [documents[idx] for idx in indices[0]]
    context = " ".join(retrieved_docs)

    prompt = f"""
    依照以下上下文回答問題：
    {context}
    問題：{query}
    """

    url = "http://grok.com?q=" + urllib.parse.quote(prompt)
    return url
    
    
    # # 準備輸入：查詢 + 檢索到的上下文
    # input_text = f"問題: {query} 上下文: {context}"
    # inputs = generator_tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True)
    
    # # 生成回答
    # outputs = generator.generate(**inputs, max_length=100, num_beams=4)
    # answer = generator_tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # return answer

def main():
    # preprocess()
    # 測試
    query = "司康有什麼口味？"
    answer = rag_query(query)
    print(f"回答: {answer}")


if __name__ == "__main__":
    # qqq
    main()
