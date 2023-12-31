import logging
import os
import time

import dotenv
from sentence_transformers import CrossEncoder

from lib.ai.loader import ModelLoader
from lib.ai.model.search import SearchResponseDto
from lib.ai.search import SearchAI


while "tests" not in os.listdir():
    os.chdir("..")
dotenv.load_dotenv()
os.environ["MONGO_DB"] = "test"
os.environ["STAGE"] = "test"
os.environ["BUCKET_KEY"] = "search-engine/test"


def test_search():
    from pykospacing import Spacing

    # given
    search_model = SearchAI(
        version="v1",
        loader=ModelLoader(
            model_dir="tests/resource/model", bucket=os.getenv("BUCKET"), bucket_key=os.getenv("BUCKET_KEY")
        ),
        cross_encoder=CrossEncoder("bongsoo/kpf-cross-encoder-v1", max_length=128),
        spacing=Spacing(),
    )
    query = "우유"
    # when
    response: SearchResponseDto = search_model.search(query)
    # then
    assert response.version == "v1"
    assert response.engine_type == "ML"
    assert len(response.results) > 0


# def test_search_parallel():
#     # given
#     search_model = SearchAI(
#             version="v1",
#             loader=ModelLoader(model_dir="tests/resource/model"),
#             cross_encoder=CrossEncoder("bongsoo/kpf-cross-encoder-v1"),
#     )
#     query = "우유"
#     # when
#     response: SearchResponseDto = search_model.search_parallel(query)
#     search_model.exit()
#     # then
#     assert response.version == "v1"
#     assert response.engine_type == "ML"
#     assert len(response.results) > 0
#
#
# def test_speed():
#     os.environ["TOKENIZERS_PARALLELISM"] = "false"
#     # given
#     search_model = SearchAI(
#             version="v1",
#             loader=ModelLoader(model_dir="tests/resource/model"),
#             cross_encoder=CrossEncoder("bongsoo/kpf-cross-encoder-v1"),
#     )
#     q1 = "우유"
#     q2 = "돼지"
#     # when
#     s = time.time()
#     search_model.search(query=q1)
#     elapsed1 = time.time() - s
#     s = time.time()
#     search_model.search_parallel(query=q2)
#     elapsed2 = time.time() - s
#     assert elapsed2 < elapsed1
#     logging.info(f"elapsed1: {elapsed1}, elapsed2: {elapsed2}")
