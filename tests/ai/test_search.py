import os

import pytest
from sentence_transformers import CrossEncoder

from lib.ai.loader import ModelLoader
from lib.ai.model.search import SearchResponseDto
from lib.ai.search import SearchAI


while "tests" not in os.listdir():
    os.chdir("..")

os.environ["MONGO_DB"] = "test"


@pytest.fixture
def models():
    return ModelLoader(model_dir="tests/resource/model").load()


def test_search(models):
    # given
    search_model = SearchAI(version="v1", models=models, cross_encoder=CrossEncoder("bongsoo/kpf-cross-encoder-v1"))
    query = "우유"
    # when
    response: SearchResponseDto = search_model.search(query)
    # then
    assert response.version == "v1"
    assert response.engine_type == "ML"
    assert len(response.results) > 0
