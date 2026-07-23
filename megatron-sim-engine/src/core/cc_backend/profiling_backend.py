"""Profile-table-driven CC backend."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from .base import CCBackend, get_backend_options
from .op_mapping import infer_collective_kind, infer_domain_dims
from .registry import register_cc_backend
from .types import CommunicationPredictionRequest


@register_cc_backend("profiling")
class ProfilingCCBackend(CCBackend):
    """Communication backend backed by a deterministic profile table.

    Expected table key format:
      "<collective_kind>|<domain_dim>|<group_size>|<data_size_bytes>"

    Example key:
      "allreduce|DP|8|268435456": 3.42
    """

    backend_name = "profiling"

    def __init__(self, simulator_config: Any) -> None:
        super().__init__(simulator_config)
        options = get_backend_options(simulator_config, self.backend_name)
        self._table = self._load_profile_table(options)

    def _load_profile_table(self, options: Dict[str, Any]) -> Dict[str, float]:
        table_obj = options.get("table")
        table_path = options.get("table_path")

        if table_obj is None and table_path:
            path = Path(table_path)
            if not path.exists():
                raise FileNotFoundError(f"profiling backend table_path not found: {path}")
            table_obj = json.loads(path.read_text(encoding="utf-8"))

        if table_obj is None:
            raise ValueError(
                "profiling backend requires `communication.backend_options.table` "
                "or `communication.backend_options.table_path`."
            )

        table: Dict[str, float] = {}
        if isinstance(table_obj, dict):
            for key, value in table_obj.items():
                table[str(key)] = float(value)
            return table

        if isinstance(table_obj, list):
            for entry in table_obj:
                if not isinstance(entry, dict):
                    raise ValueError("profiling table list entries must be dict")
                key = self._build_key(
                    collective_kind=str(entry["collective_kind"]),
                    domain_dim=str(entry["domain_dim"]),
                    group_size=int(entry["group_size"]),
                    data_size_bytes=int(entry["data_size_bytes"]),
                )
                table[key] = float(entry["duration_ms"])
            return table

        raise ValueError("profiling backend table must be dict or list")

    @staticmethod
    def _build_key(*, collective_kind: str, domain_dim: str, group_size: int, data_size_bytes: int) -> str:
        return f"{collective_kind}|{domain_dim}|{group_size}|{data_size_bytes}"

    def predict(self, request: CommunicationPredictionRequest) -> float:
        self.validate_request(request)

        collective_kind = infer_collective_kind(request.op_name)
        if request.domain_dims:
            domain_dim = request.domain_dims[0]
        else:
            domain_dim = infer_domain_dims(request.group_kind, request.op_name)[0]

        key = self._build_key(
            collective_kind=collective_kind,
            domain_dim=domain_dim,
            group_size=request.group_size,
            data_size_bytes=int(request.data_size_bytes),
        )
        if key not in self._table:
            raise KeyError(
                f"No profiling record for key {key}. "
                "Please add a profile entry or choose a different cc backend."
            )
        return float(self._table[key])
