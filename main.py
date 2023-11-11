import argparse

from fastapi import FastAPI

from lib.view import embedding, search


parser = argparse.ArgumentParser(
    prog="pyoniverse ai engine",
    description="여러 검색 모델을 앙상블하여 최선의 결과를 반환한다. 모델은 FastAPI를 통해 통신한다.",
    epilog="created by yyr",
)
# if __name__ == "__main__":
# item_search_module = SearchAI(data_path="resource/data/products.json", embedding_paths=True)
#
# print(item_search_module.ai("치즈 김밥"))
#
# while True:
#     query = input()
#     if query == "x":
#         break
#     print(item_search_module.ai(query))

app = FastAPI(
    title="pyoniverse-search-engine api",
    summary="Search Engine's API",
    version="v1",
)
app.include_router(
    search.router,
    prefix="/v1/search",
    tags=["search"],
)
app.include_router(embedding.router, prefix="/v1/embedding", tags=["embedding"])
