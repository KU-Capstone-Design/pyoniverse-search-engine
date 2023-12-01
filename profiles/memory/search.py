"""
Search AI Memory Usage 프로파일링
[이유]
- 병렬 실행으로 속도를 높이려고 했지만, 오히려 더 느려짐.
- Memory Swap에 시간이 오래 걸리는 것으로 추정(컴퓨터가 느려졌음 - 메모리가 부족할 때 나타나는 현상)
python -m memory_profiler [script].py
[Visualization]
mprof run [script].py
mprof plot -o [output].png

[Result]
Single: 최대 1.4GB까지 소비한다.
Parallel 최대 2.2GB까지 소비

[속도]
Parallel << Single
"""
from memory_profiler import profile
from sentence_transformers import CrossEncoder

from lib.ai.loader import ModelLoader
from lib.ai.search import SearchAI


# @profile
def search(query: str = "우유"):
    search_model = SearchAI(
        version="v1",
        loader=ModelLoader(model_dir="tests/resource/model"),
        cross_encoder=CrossEncoder("bongsoo/kpf-cross-encoder-v1"),
    )
    search_model.search(query)


# @profile
# def search_parallel(query: str = "우유"):
#     search_model = SearchAI(
#             version="v1",
#             loader=ModelLoader(model_dir="tests/resource/model"),
#             cross_encoder=CrossEncoder("bongsoo/kpf-cross-encoder-v1"),
#     )
#     search_model.search_parallel(query)
#
#
# if __name__ == "__main__":
#     search_parallel()
