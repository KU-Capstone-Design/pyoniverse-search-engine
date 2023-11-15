from typing import Generic, TypeVar

from pydantic import BaseModel


ResponseType = TypeVar("ResponseType", bound=BaseModel)


class ApiResponse(BaseModel, Generic[ResponseType]):
    status_code: int
    data: ResponseType
