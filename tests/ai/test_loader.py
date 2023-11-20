import os
from pathlib import Path
from typing import List

import dotenv
import pytest

from lib.ai.loader import ModelLoader
from lib.ai.model.embedding import Embedding, SearchModel


while "tests" not in os.listdir():
    os.chdir("..")
dotenv.load_dotenv()


def test_loader():
    # given
    loader = ModelLoader(model_dir="tests/resource/model")
    # when
    models: List[SearchModel] = loader.load()
    # then
    for model in models:
        assert model.type in {"sentence", "lexical"}
        assert len(model.embeddings) > 0
        assert all(isinstance(e, Embedding) for e in model.embeddings)


def test_loader_invalid_dir():
    with pytest.raises(RuntimeError):
        ModelLoader(model_dir="tests/resource/invalid")


def test_loader_empty_dir():
    empty_path = "tests/resource/empty"
    if Path(empty_path).exists():
        os.rmdir(empty_path)
    os.mkdir(empty_path)
    loader = ModelLoader(model_dir="tests/resource/empty")
    with pytest.raises(RuntimeError):
        loader.load()
