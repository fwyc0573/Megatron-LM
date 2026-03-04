#!/usr/bin/env python3
"""Run megatron-sim-engine with in-process collective-sim memoization.

This helper keeps simulator interfaces unchanged while reducing duplicate
`predict_collective_time` invocations for topology-equivalent scenarios.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
import sys
from typing import Any, Dict, Iterable, List

REPO_ROOT = Path(__file__).resolve().parents[2]
SIM_ENGINE_ROOT = REPO_ROOT / "megatron-sim-engine"
if str(SIM_ENGINE_ROOT) not in sys.path:
    sys.path.insert(0, str(SIM_ENGINE_ROOT))

from src.core.cc_backend.collective_sim_backend import CollectiveSimCCBackend


_ORIGINAL_INIT = CollectiveSimCCBackend.__init__


def _canonicalize_participants(scenario: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize participant rank IDs to topology-equivalent canonical IDs.

    Collective-sim performance depends on server-local layout, not absolute global
    rank IDs. Canonicalization greatly increases cache hit-rate for groups that are
    structurally identical but shifted in global rank space.
    """

    canonical = copy.deepcopy(scenario)
    cluster = canonical.get("cluster") or {}
    collective = canonical.get("collective") or {}
    participants: Iterable[int] = collective.get("participant_ranks") or ()
    participants = list(participants)
    if not participants:
        return canonical

    gpus_per_server = int(cluster.get("gpus_per_server", 1) or 1)
    if gpus_per_server <= 0:
        return canonical

    servers: List[int] = [int(rank) // gpus_per_server for rank in participants]
    locals_: List[int] = [int(rank) % gpus_per_server for rank in participants]

    # Preserve structural order but remove dependence on absolute server IDs.
    sorted_unique_servers = sorted(set(servers))
    server_remap = {server_id: idx for idx, server_id in enumerate(sorted_unique_servers)}
    canonical_ranks = [server_remap[sid] * gpus_per_server + local for sid, local in zip(servers, locals_)]

    canonical.setdefault("collective", {})["participant_ranks"] = canonical_ranks
    return canonical


def _patch_collective_sim_with_cache() -> None:
    def _init_with_cache(self: CollectiveSimCCBackend, simulator_config: Any) -> None:
        _ORIGINAL_INIT(self, simulator_config)

        base_predict = self.predict_collective_time
        cache: Dict[str, Dict[str, Any]] = {}
        stats = {"hits": 0, "misses": 0}

        def _cached_predict(scenario: Dict[str, Any], *, repo_root=None) -> Dict[str, Any]:
            canonical_scenario = _canonicalize_participants(scenario)
            cache_key = json.dumps(
                canonical_scenario,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
            )
            if cache_key in cache:
                stats["hits"] += 1
                return cache[cache_key]

            stats["misses"] += 1
            result = base_predict(canonical_scenario, repo_root=repo_root)
            cache[cache_key] = result
            return result

        self.predict_collective_time = _cached_predict
        self._collective_sim_prediction_cache = cache
        self._collective_sim_prediction_cache_stats = stats

    CollectiveSimCCBackend.__init__ = _init_with_cache


def main(argv: List[str]) -> int:
    _patch_collective_sim_with_cache()

    from simu_main import main as simu_main_entry

    return int(simu_main_entry(argv))


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
