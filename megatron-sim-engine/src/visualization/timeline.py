"""Visualization helpers for simulator timelines."""

from __future__ import annotations

from typing import Iterable, Optional


def render_simulation_timelines(
    simulator_engine,
    rank_start: int,
    rank_end: int,
    specific_ranks: Optional[Iterable[int]] = None,
    save_plot: bool = True,
    output_dir: str = "./log/visualization_outputs",
) -> None:
    """Render timelines through the simulator engine visualization API."""

    simulator_engine.visualize_timelines(
        wrank_id_start_end=[rank_start, rank_end],
        specific_ranks_list=list(specific_ranks) if specific_ranks is not None else None,
        save_plot=save_plot,
        output_dir=output_dir,
    )
