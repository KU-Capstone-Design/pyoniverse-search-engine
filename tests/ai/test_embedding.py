import json
import os
import re
from pathlib import Path
from typing import List

import dotenv
import pytest

from lib.ai.embedding import EmbeddingAI
from tests.util import not_raises


@pytest.fixture
def env():
    while "tests" not in os.listdir():
        os.chdir("..")
    dotenv.load_dotenv()


def test_init(env):
    # given
    with not_raises(RuntimeError):
        EmbeddingAI(embedding_dir="tests/resource/embedding")


def test_init_embedding_dir(env):
    with not_raises(RuntimeError):
        EmbeddingAI(embedding_dir="tests/resource/embedding")
        EmbeddingAI(embedding_dir="tests/resource/create-new-folder")


def test_init_embedding_dir_invalid(env):
    with pytest.raises(RuntimeError):
        EmbeddingAI(embedding_dir="tests/resource/data/products.json")


def test_preprocess_data(env):
    # given
    with open("tests/resource/data/products.json", "r") as fd:
        data = json.load(fd)
    embedding_ai = EmbeddingAI()
    id_name_map = {d["id"]: d["name"] for d in data}
    # when
    response: List[EmbeddingAI.Data] = embedding_ai.preprocess_data()
    # then
    for r in response:
        assert re.sub(r"\s", "", id_name_map[r.id]) == re.sub(r"\s", "", r.name)


def test_get_bm250k_model(env):
    # given
    embedding_ai = EmbeddingAI()
    # when
    data = embedding_ai.preprocess_data()
    model_name = embedding_ai.get_bm250k_model(data)
    # then
    model_info = embedding_ai.get_model(model_name)
    assert embedding_ai.is_lexical_model(model_name)
    assert model_info["type"] == "lexical"
    assert set(d.name for d in data) == set(c.name for c in model_info["model"][1])


def test_get_sroberta_multitask_model(env):
    # given
    embedding_ai = EmbeddingAI(embedding_dir="tests/resource/embedding")
    # when
    data = embedding_ai.preprocess_data()
    model_name = embedding_ai.get_sroberta_multitask_model(data)
    # then
    model_info = embedding_ai.get_model(model_name)
    assert embedding_ai.is_sentence_model(model_name)
    assert model_info["type"] == "sentence"
    assert set(d.id for d in data) == set(e.id for e in model_info["model"][1])


def test_sroberta_sts_model(env):
    # given
    embedding_ai = EmbeddingAI(embedding_dir="tests/resource/embedding")
    # when
    data = embedding_ai.preprocess_data()
    model_name = embedding_ai.get_sroberta_sts_model(data)
    # then
    model_info = embedding_ai.get_model(model_name)
    assert embedding_ai.is_sentence_model(model_name)
    assert model_info["type"] == "sentence"
    assert set(d.id for d in data) == set(e.id for e in model_info["model"][1])


def test_save_embedding(env):
    # given
    embedding_ai = EmbeddingAI(embedding_dir="tests/resource/embedding")
    data = embedding_ai.preprocess_data()
    model_name = embedding_ai.get_sroberta_sts_model(data)
    # when
    path = embedding_ai.save_embedding(model_name)
    # then
    assert Path(path).exists()


def test_save_embedding_invalid_model(env):
    # given
    embedding_ai = EmbeddingAI(embedding_dir="tests/resource/embedding")
    # when & then
    with pytest.raises(RuntimeError):
        embedding_ai.save_embedding("invalid")


def test_embedding_integration(env):
    # given
    embedding_ai = EmbeddingAI(embedding_dir="tests/resource/embedding")
    # when
    result = embedding_ai.execute()
    # then
    for meta in result:
        if meta.type == "lexical":
            assert embedding_ai.is_lexical_model(meta.name)
            assert Path(meta.embedding_path).exists()
        elif meta.type == "sentence":
            assert embedding_ai.is_sentence_model(meta.name)
            assert Path(meta.embedding_path).exists()
        else:
            assert False
