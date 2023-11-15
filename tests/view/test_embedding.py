import os

from fastapi.testclient import TestClient

from lib.ai.model.common import EmbeddingResponseDto
from lib.view.model.api_response import ApiResponse
from tests.view.test_model import not_raise


while "tests" not in os.listdir():
    os.chdir("..")

os.environ["STAGE"] = "test"
os.environ["MONGO_DB"] = "test"
os.environ["EMBEDDING_DIR"] = "tests/resource/embedding"


def test_post_embedding():
    from main import app

    # given
    client = TestClient(app)
    endpoint = "/test/v1/embedding/reload"
    # when
    res = client.post(endpoint)
    # then
    assert res.status_code == 200
    with not_raise():
        ApiResponse[EmbeddingResponseDto].model_validate_json(res.text, strict=True)
