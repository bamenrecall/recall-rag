import os
from typing import Any

import faiss
import numpy as np
import yaml
from generate_embeddings import get_embeddings

def add_to_faiss_index(embeddings: list[list[float]]):
    vector = np.array(embeddings)
    index = faiss.IndexFlatL2(vector.shape[1])
    index.add(vector)
    return index

def vector_search(
    index: faiss.IndexFlatL2, query_embedding: list[float], text_array: list[str], k=1
):
    distances, indices = index.search(np.array([query_embedding]), k)
    return [(text_array[i], float(dist)) for dist, i in zip(distances[0], indices[0])]

def load_data() -> dict[str, Any]:
    with open("data/embedded.yaml", "r") as file:
        return yaml.safe_load(file)

def get_contents_and_embeddings() -> tuple[list[str], list[list[float]]]:
    data = load_data()
    contents = []
    embeddings = []
    for story in data['stories']:
        contents.append(story['content'])
        embeddings.append(story['embedding'])
    return contents, embeddings

def main():
    contents, embeddings = get_contents_and_embeddings()
    faiss_index = add_to_faiss_index(embeddings)

    # search
    query_text = "深藍家庭"
    query_embedding = get_embeddings([query_text])[0].values
    search_results = vector_search(faiss_index, query_embedding, contents)
    print(f"search results for {query_text}:", search_results)
    examples = [f'{i+1}. {example}' for i, (example, _) in enumerate(search_results)]
    print(
        f"""
            繁體中文回覆
            根據以下案例生產出適合我的說服其他人的策略:
            {"\n".join(examples)}
        """,
        search_results
    )

if __name__ == "__main__":
    main()
