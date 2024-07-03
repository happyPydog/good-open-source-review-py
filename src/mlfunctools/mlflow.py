import contextlib
import functools
from typing import Any, Callable, ContextManager, ParamSpec, TypeVar, overload

import mlflow
import optuna

P = ParamSpec("P")
R = TypeVar("R")


@overload
def mlflow_run(*, run_name: str | None = None) -> ContextManager[Any]: ...


@overload
def mlflow_run(func: Callable[P, R]) -> Callable[P, R]: ...


def mlflow_run(
    func: Callable[P, R] | None = None, *, run_name: str | None = None
) -> Callable[P, R] | contextlib.AbstractContextManager[Any]:
    if func is None:
        return mlflow_context_manager(run_name)

    @functools.wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        if func is None:
            raise ValueError("Function must be provided.")
        with mlflow_context_manager():
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


def log_hyperparam_analysis(
    study: optuna.study.Study,
    params: list[str] | None = None,
) -> None: ...
