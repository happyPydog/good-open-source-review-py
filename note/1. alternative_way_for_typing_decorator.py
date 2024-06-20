import functools
import logging
import time
from typing import Callable, ParamSpec, TypeVar

P = ParamSpec("P")
R = TypeVar("R")

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def timeit(name: str | None = None) -> Callable[[Callable[P, R]], Callable[P, R]]:
    def timeit_decorator(func: Callable[P, R]) -> Callable[P, R]:
        @functools.wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            t0 = time.monotonic()
            try:
                return func(*args, **kwargs)
            finally:
                t1 = time.monotonic()
                message = f"{name or func.__name__} took {t1 - t0} seconds"
                logger.info(message)

        return wrapper

    return timeit_decorator


@timeit("Example_Function")
def example_function(n: int) -> int:
    result = 0
    for i in range(n):
        result += i * i
    return result


if __name__ == "__main__":
    example_function(1000000)
