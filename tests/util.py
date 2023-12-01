import logging
from contextlib import contextmanager
from typing import Type


@contextmanager
def not_raises(exception: Type[Exception]):
    try:
        yield
    except exception as e:
        logging.error(f"Expected Exception is raised: {repr(e)}")
        assert False
    except Exception as e:
        logging.error(f"Unexpected Exception is raised: {repr(e)}")
        assert False
    else:
        assert True
