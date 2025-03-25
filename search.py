import urllib.parse
from typing import Any

import faiss
import numpy as np
import yaml

from pre_process import get_embeddings


def gen_faiss(embeddings: list[list[float]]):
    vector = np.array(embeddings)
    index = faiss.IndexFlatL2(vector.shape[1])
    index.add(vector)
    return index


def vector_search(
    index: faiss.IndexFlatL2, query_embedding: list[float], summaries: list[str], k=2
):
    distances, indices = index.search(np.array([query_embedding]), k)
    return [(summaries[i], float(dist)) for dist, i in zip(distances[0], indices[0])]


def load_data() -> dict[str, Any]:
    with open("data/processed.yaml", "r") as file:
        return yaml.safe_load(file)


def get_contents_and_embeddings() -> tuple[list[str], list[list[float]]]:
    data = load_data()

    embeddings = []
    summaries = []
    for story in data["stories"]:
        embeddings.append(story["embedding"])
        summaries.append(story["summary"])
    return embeddings, summaries


def main():
    embeddings, summaries = get_contents_and_embeddings()
    faiss_index = gen_faiss(embeddings)

    # search
    query_text = "我爸媽都投國民黨，我要怎麼說服他們？"
    query_embedding = get_embeddings([query_text])[0]
    search_results = vector_search(faiss_index, query_embedding, summaries)

    print(f"search results for {query_text}:", search_results)
    examples = [f"{i+1}. {example}" for i, (example, _) in enumerate(search_results)]
    prompt = f"""
        繁體中文回覆
        根據以下案例生產出適合我的說服其他人的策略，並給出聊天範例：
        {"\n".join(examples)}
    """
    url = "http://grok.com?q=" + urllib.parse.quote(prompt)
    print(url)


if __name__ == "__main__":
    main()
