import os
from typing import List

import dotenv
import pytest


while "tests" not in os.listdir():
    os.chdir("..")
dotenv.load_dotenv()


def test_loader_sentence():
    # given
    loader = Loader(model_path="tests/resource/model")
    # when
    sentence_models: List[Model] = loader.load(type="sentence")
    # then
    for model in sentence_models:
        assert model.type == "sentence"
        assert len(model.embeddings) > 0
        assert all(isinstance(e) for e in model.embeddings)


def test_loader_lexical():
    # given
    loader = Loader(model_path="tests/resource/model")
    # when
    sentence_models: List[Model] = loader.load(type="lexical")
    # then
    for model in sentence_models:
        assert model.type == "lexical"
        assert len(model.embeddings) > 0
        assert all(isinstance(e) for e in model.embeddings)


def test_loader_not_found():
    # given
    loader = Loader(model_path="tests/resource/invalid")
    # when & then
    with pytest.raises(FileNotFoundError):
        loader.load(type="lexical")
