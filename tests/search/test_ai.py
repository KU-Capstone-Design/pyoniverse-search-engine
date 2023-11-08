import json
import os
import re
from pathlib import Path
from typing import List

import pytest

from lib.ai.search import ItemSearchAI


@pytest.fixture
def env():
    while "tests" not in os.listdir():
        os.chdir("..")


def test_init(env):
    # given
    search_ai = ItemSearchAI(data_path="tests/resource/data/products.json")
    assert True


def test_init_invalid_data_path(env):
    # given
    try:
        # Invalid Path
        ItemSearchAI(data_path="tests/resource/data/invalid-path.json")
        assert False
    except RuntimeError:
        assert True

    try:
        # Directory Path
        ItemSearchAI(data_path="tests/resource/data")
        assert False
    except RuntimeError:
        assert True

    try:
        # Not Json
        ItemSearchAI(data_path="tests/resource/data/products.csv")
        assert False
    except RuntimeError:
        assert True


def test_init_embedding_dir(env):
    try:
        ItemSearchAI(embedding_dir="tests/resource/embedding")
        assert True
    except Exception:
        assert False


def test_init_embedding_dir_invalid(env):
    try:
        ItemSearchAI(embedding_dir="tests/resource/data/products.json")
        assert False
    except Exception:
        assert True

    try:
        ItemSearchAI(embedding_dir="tests/resource/invalid")
        assert False
    except Exception:
        assert True


def test_preprocess_data(env):
    # given
    with open("tests/resource/data/products.json", "r") as fd:
        data = json.load(fd)
    search_ai = ItemSearchAI(data_path="tests/resource/data/products.json")
    id_name_map = {d["id"]: d["name"] for d in data}
    # when
    response: List[dict] = search_ai.preprocess_data()
    # then
    for r in response:
        assert re.sub(r"\s", "", id_name_map[r["id"]]) == re.sub(r"\s", "", r["product"])


def test_preprocess_data_empty_file(env):
    # given
    search_ai = ItemSearchAI(data_path="tests/resource/data/empty-products.json")
    # when & then
    try:
        search_ai.preprocess_data()
        assert False
    except RuntimeError:
        assert True


def test_get_bm250k_model(env):
    # given
    search_ai = ItemSearchAI(data_path="tests/resource/data/products.json")
    # when
    data = search_ai.preprocess_data()
    model_name = search_ai.get_bm250k_model(data)
    # then
    assert search_ai.is_lexical_model(model_name)
    assert set(d["name"] for d in data) == set(c["name"] for c in search_ai.lexical_models[model_name][1])


def test_get_sroberta_multitask_model(env):
    # given
    search_ai = ItemSearchAI(data_path="tests/resource/data/products.json")
    # when
    data = search_ai.preprocess_data()
    model_name = search_ai.get_sroberta_multitask_model(data)
    # then
    assert search_ai.is_sentence_model(model_name)
    assert set(d["id"] for d in data) == set(e["id"] for e in search_ai.sentence_models[model_name][1])


def test_sroberta_sts_model(env):
    # given
    search_ai = ItemSearchAI(data_path="tests/resource/data/products.json")
    # when
    data = search_ai.preprocess_data()
    model_name = search_ai.get_sroberta_sts_model(data)
    # then
    assert search_ai.is_sentence_model(model_name)
    assert set(d["id"] for d in data) == set(e["id"] for e in search_ai.sentence_models[model_name][1])


def test_save_embedding(env):
    # given
    search_ai = ItemSearchAI(data_path="tests/resource/data/products.json", embedding_dir="tests/resource/embedding")
    data = search_ai.preprocess_data()
    model_name = search_ai.get_sroberta_sts_model(data)
    # when
    path = search_ai.save_embedding(model_name)
    # then
    assert Path(path).exists()


def test_save_embedding_invalid_model(env):
    # given
    search_ai = ItemSearchAI(data_path="tests/resource/data/products.json", embedding_dir="tests/resource/embedding")
    # when & then
    try:
        search_ai.save_embedding("invalid")
        assert False
    except RuntimeError:
        assert True
