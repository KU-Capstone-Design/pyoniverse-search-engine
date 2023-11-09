import json
import os
import re
from pathlib import Path
from typing import List

import pytest

from lib.ai.embedding import EmbeddingAI


@pytest.fixture
def env():
    while "tests" not in os.listdir():
        os.chdir("..")


def test_init(env):
    # given
    try:
        embedding_ai = EmbeddingAI(
            data_path="tests/resource/data/products.json", embedding_dir="tests/resource/embedding"
        )
        assert True
    except Exception:
        assert False


def test_init_invalid_data_path(env):
    # given
    try:
        # Invalid Path
        EmbeddingAI(data_path="tests/resource/data/invalid-path.json", embedding_dir="tests/resource/embedding")
        assert False
    except RuntimeError:
        assert True

    try:
        # Directory Path
        EmbeddingAI(data_path="tests/resource/data", embedding_dir="tests/resource/embedding")
        assert False
    except RuntimeError:
        assert True

    try:
        # Not Json
        EmbeddingAI(data_path="tests/resource/data/products.csv", embedding_dir="tests/resource/embedding")
        assert False
    except RuntimeError:
        assert True


def test_init_embedding_dir(env):
    try:
        EmbeddingAI(data_path="tests/resource/data/products.json", embedding_dir="tests/resource/embedding")
        assert True
    except Exception:
        assert False


def test_init_embedding_dir_invalid(env):
    try:
        EmbeddingAI(data_path="tests/resource/data/products.json", embedding_dir="tests/resource/data/products.json")
        assert False
    except Exception:
        assert True

    try:
        EmbeddingAI(data_path="tests/resource/data/products.json", embedding_dir="tests/resource/invalid")
        assert False
    except Exception:
        assert True


def test_preprocess_data(env):
    # given
    with open("tests/resource/data/products.json", "r") as fd:
        data = json.load(fd)
    embedding_ai = EmbeddingAI(data_path="tests/resource/data/products.json")
    id_name_map = {d["id"]: d["name"] for d in data}
    # when
    response: List[dict] = embedding_ai.preprocess_data()
    # then
    for r in response:
        assert re.sub(r"\s", "", id_name_map[r["id"]]) == re.sub(r"\s", "", r["name"])


def test_preprocess_data_empty_file(env):
    # given
    embedding_ai = EmbeddingAI(data_path="tests/resource/data/empty-products.json")
    # when & then
    try:
        embedding_ai.preprocess_data()
        assert False
    except RuntimeError:
        assert True


def test_get_bm250k_model(env):
    # given
    embedding_ai = EmbeddingAI(data_path="tests/resource/data/products.json")
    # when
    data = embedding_ai.preprocess_data()
    model_name = embedding_ai.get_bm250k_model(data)
    # then
    model_info = embedding_ai.get_model(model_name)
    assert embedding_ai.is_lexical_model(model_name)
    assert model_info["type"] == "lexical"
    assert set(d["name"] for d in data) == set(c["name"] for c in model_info["model"][1])


def test_get_sroberta_multitask_model(env):
    # given
    embedding_ai = EmbeddingAI(data_path="tests/resource/data/products.json")
    # when
    data = embedding_ai.preprocess_data()
    model_name = embedding_ai.get_sroberta_multitask_model(data)
    # then
    model_info = embedding_ai.get_model(model_name)
    assert embedding_ai.is_sentence_model(model_name)
    assert model_info["type"] == "sentence"
    assert set(d["id"] for d in data) == set(e["id"] for e in model_info["model"][1])


def test_sroberta_sts_model(env):
    # given
    embedding_ai = EmbeddingAI(data_path="tests/resource/data/products.json")
    # when
    data = embedding_ai.preprocess_data()
    model_name = embedding_ai.get_sroberta_sts_model(data)
    # then
    model_info = embedding_ai.get_model(model_name)
    assert embedding_ai.is_sentence_model(model_name)
    assert model_info["type"] == "sentence"
    assert set(d["id"] for d in data) == set(e["id"] for e in model_info["model"][1])


def test_save_embedding(env):
    # given
    embedding_ai = EmbeddingAI(data_path="tests/resource/data/products.json", embedding_dir="tests/resource/embedding")
    data = embedding_ai.preprocess_data()
    model_name = embedding_ai.get_sroberta_sts_model(data)
    # when
    path = embedding_ai.save_embedding(model_name)
    # then
    assert Path(path).exists()


def test_save_embedding_invalid_model(env):
    # given
    embedding_ai = EmbeddingAI(data_path="tests/resource/data/products.json", embedding_dir="tests/resource/embedding")
    # when & then
    try:
        embedding_ai.save_embedding("invalid")
        assert False
    except RuntimeError:
        assert True


def test_embedding_integration(env):
    # given
    embedding_ai = EmbeddingAI(data_path="tests/resource/data/products.json", embedding_dir="tests/resource/embedding")
    # when
    result = embedding_ai.execute()
    # then
    for key, val in result.items():
        if key == "lexical":
            for model_name, embedding_saved_path in val.items():
                assert embedding_ai.is_lexical_model(model_name)
                assert Path(embedding_saved_path).exists()
        elif key == "sentence":
            for model_name, embedding_saved_path in val.items():
                assert embedding_ai.is_sentence_model(model_name)
                assert Path(embedding_saved_path).exists()
        else:
            assert False
