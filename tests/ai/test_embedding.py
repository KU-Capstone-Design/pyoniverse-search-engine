import os
import pickle
import re
from pathlib import Path
from typing import List

import dotenv
import pytest
from pymongo import MongoClient, ReadPreference
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from lib.ai.builder import ModelBuilder
from lib.ai.model.embedding import EmbeddingResponseDto, SearchModel
from tests.util import not_raises


while "tests" not in os.listdir():
    os.chdir("..")
dotenv.load_dotenv()
os.environ["MONGO_DB"] = "test"


def test_init():
    # given
    with not_raises(RuntimeError):
        ModelBuilder(db_uri=os.getenv("MONGO_URI"), db_name=os.getenv("MONGO_DB"), model_dir="tests/resource/model")
        ModelBuilder(
            db_uri=os.getenv("MONGO_URI"), db_name=os.getenv("MONGO_DB"), model_dir="tests/resource/create-new-folder"
        )
    os.rmdir("tests/resource/create-new-folder")

    with pytest.raises(RuntimeError):
        ModelBuilder(db_name=os.getenv("MONGO_DB"), model_dir="tests/resource/model")
        ModelBuilder(db_uri=os.getenv("MONGO_URI"), model_dir="tests/resource/model")


def test_preprocess_data():
    # given
    builder = ModelBuilder(
        db_uri=os.getenv("MONGO_URI"), db_name=os.getenv("MONGO_DB"), model_dir="tests/resource/model"
    )
    db = MongoClient(os.getenv("MONGO_URI")).get_database(
        os.getenv("MONGO_DB"), read_preference=ReadPreference.SECONDARY_PREFERRED
    )
    data = db["products"].find(projection={"id": True, "name": True, "_id": False})
    id_name_map = {d["id"]: d["name"] for d in data}
    # when
    response: List[ModelBuilder.Data] = builder.preprocess_data()
    # then
    for r in response:
        assert re.sub(r"\s", "", id_name_map[r.id]) == re.sub(r"\s", "", r.name)


def test_make_bm250k_model():
    # given
    builder = ModelBuilder(
        db_uri=os.getenv("MONGO_URI"), db_name=os.getenv("MONGO_DB"), model_dir="tests/resource/model"
    )
    # when
    data = builder.preprocess_data()
    model_name = builder.make_bm250k_model(data)
    # then
    model_info = builder.get_model(model_name)
    assert builder.is_lexical_model(model_name)
    assert model_info["type"] == "lexical"
    assert set(d.name for d in data) == set(c.name for c in model_info["model"][1])


def test_make_sroberta_multitask_model():
    # given
    builder = ModelBuilder(
        db_uri=os.getenv("MONGO_URI"), db_name=os.getenv("MONGO_DB"), model_dir="tests/resource/model"
    )
    # when
    data = builder.preprocess_data()
    model_name = builder.make_sroberta_multitask_model(data)
    # then
    model_info = builder.get_model(model_name)
    assert builder.is_sentence_model(model_name)
    assert model_info["type"] == "sentence"
    assert set(d.id for d in data) == set(e.id for e in model_info["model"][1])


def test_make_sroberta_sts_model():
    # given
    builder = ModelBuilder(
        db_uri=os.getenv("MONGO_URI"), db_name=os.getenv("MONGO_DB"), model_dir="tests/resource/model"
    )
    # when
    data = builder.preprocess_data()
    model_name = builder.make_sroberta_sts_model(data)
    # then
    model_info = builder.get_model(model_name)
    assert builder.is_sentence_model(model_name)
    assert model_info["type"] == "sentence"
    assert set(d.id for d in data) == set(e.id for e in model_info["model"][1])


def test_save_model():
    # given
    builder = ModelBuilder(
        db_uri=os.getenv("MONGO_URI"), db_name=os.getenv("MONGO_DB"), model_dir="tests/resource/model"
    )
    data = builder.preprocess_data()
    model_name = builder.make_sroberta_sts_model(data)
    # when
    path = builder.save(model_name)
    # then
    assert Path(path).exists()

    with open(path, "rb") as fd:
        search_model: SearchModel = pickle.load(fd)
    assert isinstance(search_model, SearchModel)
    assert len(search_model.embeddings) > 0
    assert search_model.type == "sentence"
    assert isinstance(search_model.engine, SentenceTransformer)


def test_save_invalid_model():
    # given
    builder = ModelBuilder(
        db_uri=os.getenv("MONGO_URI"), db_name=os.getenv("MONGO_DB"), model_dir="tests/resource/model"
    )
    # when & then
    with pytest.raises(RuntimeError):
        builder.save("invalid")


def test_builder_integration():
    # given
    builder = ModelBuilder(
        db_uri=os.getenv("MONGO_URI"), db_name=os.getenv("MONGO_DB"), model_dir="tests/resource/model"
    )
    # when
    results: EmbeddingResponseDto = builder.execute()
    # then
    for meta in results.models:
        if meta.type == "lexical":
            assert builder.is_lexical_model(meta.name)
            assert Path(meta.model_path).exists()
            with open(meta.model_path, "rb") as fd:
                search_model = pickle.load(fd)
                assert isinstance(search_model.engine, BM25Okapi)
        elif meta.type == "sentence":
            assert builder.is_sentence_model(meta.name)
            assert Path(meta.model_path).exists()
            with open(meta.model_path, "rb") as fd:
                search_model = pickle.load(fd)
                assert isinstance(search_model.engine, SentenceTransformer)
        else:
            assert False
