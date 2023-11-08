from lib.ai.search import ItemSearchAI


if __name__ == "__main__":
    print("make embedding")
    print("start test")
    item_search_module = ItemSearchAI(data_path="resource/data/products.json", embedding_paths=True)

    print(item_search_module.search("치즈 김밥"))

    while True:
        query = input()
        if query == "x":
            break
        print(item_search_module.search(query))
