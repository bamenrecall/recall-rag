import os
from typing import Any

import numpy as np
from google import genai
import yaml

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=GEMINI_API_KEY)


def get_embeddings(texts: list[str]) -> list[genai.types.ContentEmbedding]:
    response = client.models.embed_content(
        model="gemini-embedding-exp-03-07",
        contents=texts,
        config={"output_dimensionality": 64},
    )
    return response.embeddings


def load_data() -> dict[str, Any]:
    with open("data/raw.yaml", "r") as file:
        return yaml.safe_load(file)


def save_data(data: dict[str, Any]) -> None:
    with open("data/embedded.yaml", "w") as file:
        yaml.dump(data, file, allow_unicode=True, default_flow_style=True)


def main():
    data = load_data()

    contents = [story["content"] for story in data["stories"]]
    embeddings = get_embeddings(contents)

    embedding_values = [np.array(emb.values).tolist() for emb in embeddings]
    embedded_stories = [
        {**story, "embedding": embedding}
        for story, embedding in zip(data["stories"], embedding_values)
    ]

    new_data = {"stories": embedded_stories}
    save_data(new_data)


if __name__ == "__main__":
    main()
