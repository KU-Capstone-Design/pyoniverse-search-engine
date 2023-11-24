"""
Search AI Time Profiling
[이유]
병렬 실행시 실행 속도 확인
kernprof -l -v -u 1e-3 -o profiles/time/report/profile_search.txt profiles/time/profile_search.py (밀리초)

[Result]

"""
import os
import sys


sys.path.append(os.getcwd())
from line_profiler import profile
from sentence_transformers import CrossEncoder

from lib.ai.loader import ModelLoader
from lib.ai.search import SearchAI


search_model = SearchAI(
    version="v1",
    loader=ModelLoader(model_dir="tests/resource/model"),
    cross_encoder=CrossEncoder("bongsoo/kpf-cross-encoder-v1"),
)


# @profile
def search(query: str = "떡"):
    return search_model.search(query)


# @profile
# def search_parallel(query: str = "밥"):
#     return search_model.search_parallel(query)


# if __name__ == "__main__":
#     res = search_parallel()
#     # res = search()
