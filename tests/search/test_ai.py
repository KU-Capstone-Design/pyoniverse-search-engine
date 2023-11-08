import json
import os
import re
from typing import List

import pytest

from lib.search.ai import ItemSearchAI


@pytest.fixture
def env():
    while "tests" not in os.listdir():
        os.chdir("..")


def test_init(env):
    # given
    search_ai = ItemSearchAI(data_path="tests/data/products.json")
    assert True


def test_init_invalid_data_path(env):
    # given
    try:
        # Invalid Path
        ItemSearchAI(data_path="tests/data/invalid-path.json")
        assert False
    except RuntimeError:
        assert True

    try:
        # Directory Path
        ItemSearchAI(data_path="tests/data")
        assert False
    except RuntimeError:
        assert True

    try:
        # Not Json
        ItemSearchAI(data_path="tests/data/products.csv")
        assert False
    except RuntimeError:
        assert True


def test_preprocess_data(env):
    # given
    with open("tests/data/products.json", "r") as fd:
        data = json.load(fd)
    search_ai = ItemSearchAI(data_path="tests/data/products.json")
    id_name_map = {d["id"]: d["name"] for d in data}
    # when
    response: List[dict] = search_ai.preprocess_data()
    # then
    for r in response:
        assert re.sub(r"\s", "", id_name_map[r["id"]]) == re.sub(r"\s", "", r["product"])


def test_preprocess_data_empty_file(env):
    # given
    search_ai = ItemSearchAI(data_path="tests/data/empty-products.json")
    # when & then
    try:
        search_ai.preprocess_data()
        assert False
    except RuntimeError:
        assert True


def test_get_bm250k_model(env):
    # given
    search_ai = ItemSearchAI(data_path="tests/data/products.json")
    # when
    data = search_ai.preprocess_data()
    lexical_model, corpus = search_ai.get_bm250k_model(data)
    # then
    assert lexical_model is search_ai.get_bm250k_model(data)[0]
    assert set(corpus) == set(d["name"] for d in data)


def test_get_sroberta_multitask_model(env):
    # given
    search_ai = ItemSearchAI(data_path="tests/data/products.json")
    # when
    data = search_ai.preprocess_data()
    sentence_model, embedding = search_ai.get_sroberta_multitask_model(data)
    # then
    assert (sentence_model, embedding) is search_ai.get_bm250k_model(data)
    assert set(d["id"] for d in data) == set(e["id"] for e in embedding)


def test_sroberta_sts_model(env):
    # given
    search_ai = ItemSearchAI(data_path="tests/data/products.json")
    # when
    data = search_ai.preprocess_data()
    sentence_model, embedding = search_ai.get_sroberta_sts_model(data)
    # then
    assert (sentence_model, embedding) is search_ai.get_bm250k_model(data)
    assert set(d["id"] for d in data) == set(e["id"] for e in embedding)
