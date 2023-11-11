import argparse


parser = argparse.ArgumentParser(
    prog="pyoniverse ai engine",
    description="여러 검색 모델을 앙상블하여 최선의 결과를 반환한다. 모델은 FastAPI를 통해 통신한다.",
    epilog="created by yyr",
)
if __name__ == "__main__":
    print("make embedding")
    print("start test")
    # item_search_module = SearchAI(data_path="resource/data/products.json", embedding_paths=True)
    #
    # print(item_search_module.ai("치즈 김밥"))
    #
    # while True:
    #     query = input()
    #     if query == "x":
    #         break
    #     print(item_search_module.ai(query))
