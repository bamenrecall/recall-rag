from fastapi import FastAPI
from contextlib import asynccontextmanager
from faiss import IndexFlatL2
from search import get_contents_and_embeddings, gen_faiss, get_prompt_url

ml_models = {}

def init_faiss() -> IndexFlatL2:
    embeddings, summaries = get_contents_and_embeddings()
    faiss_index = gen_faiss(embeddings)
    return faiss_index

@asynccontextmanager
async def lifespan(app: FastAPI):
    ml_models["faiss"] = init_faiss()
    yield
    ml_models.clear()


app = FastAPI(lifespan=lifespan)

@app.get("/health")
async def health_check():
    return {"status": "ok"}

@app.get("/search")
async def search(q: str | None = None):
    return {
        "query": q,
        "url": get_prompt_url(q, ml_models["faiss"])
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
