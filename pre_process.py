import os
from typing import Any

import numpy as np
import yaml
from google import genai
from google.genai import types

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=GEMINI_API_KEY)


def get_embeddings(texts: list[str]) -> list[list[float]]:
    response = client.models.embed_content(
        model="gemini-embedding-exp-03-07",
        contents=texts,
        config={"output_dimensionality": 64},
    )

    return [np.array(emb.values).tolist() for emb in response.embeddings]


def get_summary(text: str) -> str:
    contents = [
        types.Content(
            role="user",
            parts=[
                types.Part.from_text(
                    text=f"""
                    用列點的方式總結他為什麼可以說服他的朋友簽署罷免連署書

                    {text}
                """
                ),
            ],
        )
    ]

    response = ""
    for chunk in client.models.generate_content_stream(
        model="gemini-2.0-flash-thinking-exp-01-21",
        contents=contents,
        config=types.GenerateContentConfig(
            temperature=2,
            top_p=0.95,
            top_k=64,
            max_output_tokens=65536,
            response_mime_type="text/plain",
        ),
    ):
        response += chunk.text
    return response


def load_data() -> dict[str, Any]:
    with open("data/raw.yaml", "r") as file:
        return yaml.safe_load(file)


def save_data(data: dict[str, Any]) -> None:
    with open("data/processed.yaml", "w") as file:
        yaml.dump(data, file, allow_unicode=True, default_flow_style=True)


def main():
    data = load_data()
    stories = data["stories"]
    sources = [story["source"] for story in stories]
    contents = [story["content"] for story in stories]
    summaries = [get_summary(content) for content in contents]
    embeddings = get_embeddings(contents)

    processed_stories = [
        {
            "source": source,
            "content": content,
            "embedding": embedding,
            "summary": summary,
        }
        for source, content, embedding, summary in zip(
            sources, contents, embeddings, summaries
        )
    ]

    new_data = {"stories": processed_stories}
    save_data(new_data)


if __name__ == "__main__":
    main()
