from fastapi import FastAPI
from fastapi.responses import JSONResponse
from scone import rag_query

app = FastAPI()

@app.get("/health")
async def health_check():
    return {"status": "ok"}

@app.get("/search")
async def search(q: str | None = None):
    return {
        "query": q,
        "url": rag_query(q)
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
