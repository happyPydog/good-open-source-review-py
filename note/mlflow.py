import contextlib
import functools
from typing import Any, Callable, ParamSpec, TypeVar

import mlflow

P = ParamSpec("P")
R = TypeVar("R")


def mlflow_run(
    func: Callable[P, R] | None = None, *, run_name: str | None = None
) -> Callable[P, R] | contextlib.AbstractContextManager[Any]:
    if func is None:
        return mlflow_context_manager(run_name)

    if callable(func):

        @functools.wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> Any:
            with mlflow_context_manager(run_name):
                return func(*args, **kwargs)

        return wrapper


@contextlib.contextmanager
def mlflow_context_manager(run_name: str | None = None):
    """Context manager for handling MLflow runs."""
    run = mlflow.start_run(run_name=run_name)
    try:
        yield run
    finally:
        mlflow.end_run()
