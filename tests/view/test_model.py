import logging
import traceback
from contextlib import contextmanager

import pytest
from pydantic import BaseModel, ValidationError

from lib.view.model.api_response import ApiResponse


@contextmanager
def not_raise():
    try:
        yield
    except Exception as e:
        logging.error(traceback.format_exc())
        assert False
    else:
        assert True


def test_api():
    # given
    class SimpleResponse(BaseModel):
        result: int

    class OtherResponse(BaseModel):
        result: int

    # when & then
    with not_raise():
        ApiResponse[SimpleResponse](status_code=200, data=SimpleResponse(result=3))
    with pytest.raises(ValidationError):
        ApiResponse[SimpleResponse](status_code=200, data=OtherResponse(result=3))
