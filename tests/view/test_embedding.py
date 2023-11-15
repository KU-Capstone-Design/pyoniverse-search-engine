import os

from fastapi.testclient import TestClient


while "tests" not in os.listdir():
    os.chdir("..")

os.environ["STAGE"] = "test"


def test_post_embedding():
    from main import app

    # given
    client = TestClient(app)
    endpoint = "/test/v1/embedding/reload"
    # when
    res = client.post(endpoint)
    # then
    assert res.status_code == 200
