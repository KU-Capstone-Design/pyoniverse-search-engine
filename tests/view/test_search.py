import os

from fastapi.testclient import TestClient

from lib.ai.model.search import SearchResponseDto
from lib.view.model.api_response import ApiResponse
from tests.view.test_model import not_raise


while "tests" not in os.listdir():
    os.chdir("..")

os.environ["STAGE"] = "test"
os.environ["MODEL_DIR"] = "tests/resource/model"


def test_search():
    from main import app

    # given
    client = TestClient(app)
    url = "/test/v1/search"
    query = "우유"
    # when
    res = client.get(f"{url}/{query}")
    # then
    assert res.status_code == 200
    with not_raise():
        ApiResponse[SearchResponseDto].model_validate_json(res.text, strict=True)


def test_search_empty():
    from main import app

    # given
    client = TestClient(app)
    url = "/test/v1/search"
    query = "이것은 빈 검색 결과를 반환해야 합니다."
    # when
    res = client.get(f"{url}/{query}")
    # then
    assert res.status_code == 404
