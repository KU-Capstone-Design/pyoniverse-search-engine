import json
import os
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


def test_preprocess_data():
    # given
    with open("tests/data/products.json", "r") as fd:
        data = json.load(fd)
    search_ai = ItemSearchAI(data_path="tests/data/products.json")
    id_name_map = {d["id"]: d["name"] for d in data}
    # when
    response: List[dict] = search_ai._preprocess_data()
    # then
    for r in response:
        assert id_name_map[r["id"]] == id_name_map["product"]


def test_preprocess_data_invalid_path():
    # given
    search_ai = ItemSearchAI(data_path="tests/data/invalid-path.json")
    # when & then
    try:
        search_ai._preprocess_data()
        assert False
    except RuntimeError:
        assert True


def test_preprocess_data_invalid_data_format():
    # given
    search_ai = ItemSearchAI(data_path="tests/data/invalid-products.json")
    # when & then
    try:
        search_ai._preprocess_data()
        assert False
    except RuntimeError:
        assert True


def test_preprocess_data_empty_file():
    # given
    search_ai = ItemSearchAI(data_path="tests/data/empty-products.json")
    # when & then
    try:
        search_ai._preprocess_data()
        assert False
    except RuntimeError:
        assert True
