import os

import faiss
import numpy as np
from google import genai

from data import text_array

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

client = genai.Client(api_key=GEMINI_API_KEY)


def add_to_faiss_index(embeddings: list[genai.types.ContentEmbedding]):
    vector = np.array([embedding.values for embedding in embeddings])
    index = faiss.IndexFlatL2(vector.shape[1])
    index.add(vector)
    return index


def vector_search(
    index: faiss.IndexFlatL2, query_embedding: list[float], text_array: list[str], k=1
):
    distances, indices = index.search(np.array([query_embedding]), k)
    return [(text_array[i], float(dist)) for dist, i in zip(distances[0], indices[0])]


def get_embeddings(texts: list[str]) -> list[genai.types.ContentEmbedding]:
    response = client.models.embed_content(
        model="gemini-embedding-exp-03-07",
        contents=texts,
        config={"output_dimensionality": 64},
    )
    return response.embeddings


def main():
    embedding_array = get_embeddings(text_array)

    faiss_index = add_to_faiss_index(embedding_array)

    # 輸入問題來搜譣
    query_text = "罷免"
    query_embedding = get_embeddings([query_text])[0].values
    search_results = vector_search(faiss_index, query_embedding, text_array)
    print(f"尋找 {query_text}:", search_results)


if __name__ == "__main__":
    main()
