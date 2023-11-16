from fastapi import FastAPI

from lib.config import Settings
from lib.view import embedding, search


settings = Settings()

app = FastAPI(
    title=settings.app_name,
    summary=settings.summary,
    version=settings.version,
)
app.include_router(
    search.router,
    prefix=f"/{settings.stage}/{settings.version}/search",
    tags=["search"],
)
app.include_router(embedding.router, prefix=f"/{settings.stage}/{settings.version}/embedding", tags=["embedding"])


@app.get("/ping")
def ping() -> str:
    return "ok"
