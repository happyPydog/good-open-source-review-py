"""Types"""

from typing import Any, Callable

import optuna
import optuna.study.study

ArrayLike = Any

OptunaCallbackFuncType = Callable[
    [optuna.study.Study, optuna.trial.FrozenTrial], None
]
