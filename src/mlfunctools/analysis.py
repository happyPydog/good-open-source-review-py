"""Anaylsis functions tools."""

from typing import Any

import optuna
from optuna.visualization import (
    plot_contour,
    plot_edf,
    plot_optimization_history,
    plot_parallel_coordinate,
    plot_param_importances,
    plot_rank,
    plot_slice,
    plot_timeline,
)
from plotly import graph_objects as go

vis_funcs: dict[str, Any] = {
    "plot_contour": plot_contour,
    "plot_edf": plot_edf,
    # "plot_intermediate_values": plot_intermediate_values, # ! Need to setup pruning feature, so comment out
    "plot_optimization_history": plot_optimization_history,
    "plot_parallel_coordinate": plot_parallel_coordinate,
    "plot_param_importances": plot_param_importances,
    "plot_rank": plot_rank,
    "plot_slice": plot_slice,
    "plot_timeline": plot_timeline,
}

_has_params_vis_funcs = {
    "plot_contour",
    "plot_parallel_coordinate",
    "plot_param_importances",
    "plot_slice",
    "plot_rank",
}


def analysis_param(
    study: optuna.study.Study,
    vis_funcs: dict[str, Any] = vis_funcs,
    params: list[str] | None = None,
    display: bool = True,
) -> list[go.Figure]:
    """Visualize the parameter analysis."""
    figs = []
    for key, func in vis_funcs.items():
        if key in _has_params_vis_funcs and params is not None:
            fig = func(study, params=params)
        else:
            fig = func(study)

        if display:
            fig.show()

        figs.append(fig)

    return figs
