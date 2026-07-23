"""collective-sim CC backend wrapper."""

from __future__ import annotations

import atexit
import json
import sys
from bisect import bisect_left
from dataclasses import dataclass
import math
from pathlib import Path
import re
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

from .base import CCBackend, get_backend_options
from .op_mapping import infer_collective_kind, infer_domain_dims
from .registry import register_cc_backend
from .types import CommunicationPredictionRequest

_SUPPORTED_DOMAIN_DIMS = {"TP", "CP", "DP", "EP"}

_GPU_PROFILE_DEFAULTS: Dict[str, Dict[str, Any]] = {
    "h100_sxm": {
        "intra_server": {
            "model": "nvlink_analytic",
            "nvlink_one_way_bw_GBps": 450.0,
            "nvlink_latency_us": 0.5,
            "nvlink_efficiency": 0.8,
        },
        "network": {"linkspeed_mbps": 400000, "mtu": 9216, "q": 64, "cwnd": 64},
        "topology_kind": "rail",
        "topology_spines": 8,
        "topology_paths": 8,
    },
    "h800_sxm": {
        "intra_server": {
            "model": "nvlink_analytic",
            "nvlink_one_way_bw_GBps": 200.0,
            "nvlink_latency_us": 0.6,
            "nvlink_efficiency": 0.8,
        },
        "network": {"linkspeed_mbps": 400000, "mtu": 9216, "q": 64, "cwnd": 64},
        "topology_kind": "rail",
        "topology_spines": 8,
        "topology_paths": 8,
    },
    "a100_sxm": {
        "intra_server": {
            "model": "nvlink_analytic",
            "nvlink_one_way_bw_GBps": 300.0,
            "nvlink_latency_us": 0.8,
            "nvlink_efficiency": 0.78,
        },
        "network": {"linkspeed_mbps": 200000, "mtu": 9216, "q": 48, "cwnd": 32},
        "topology_kind": "rail",
        "topology_spines": 8,
        "topology_paths": 8,
    },
    "a800_sxm": {
        "intra_server": {
            "model": "nvlink_analytic",
            "nvlink_one_way_bw_GBps": 180.0,
            "nvlink_latency_us": 0.9,
            "nvlink_efficiency": 0.76,
        },
        "network": {"linkspeed_mbps": 200000, "mtu": 9216, "q": 48, "cwnd": 32},
        "topology_kind": "rail",
        "topology_spines": 8,
        "topology_paths": 8,
    },
}


_SENDRECV_LINE_PATTERN = re.compile(
    r"^\s*(?P<size>\d+)\s+\d+\s+\w+\s+\w+\s+\w+\s+(?P<time_us>[0-9]+(?:\.[0-9]+)?)\s+"
)


@dataclass(frozen=True)
class _P2PTablePoint:
    size_bytes: int
    latency_ms: float


class _P2PMeasuredTable:
    """Measured sendrecv curve for intra/inter node latency interpolation."""

    def __init__(self, profile_dir: Path) -> None:
        self.profile_dir = profile_dir
        self.intra_points = self._load_points("single_node_sendrecv_*.txt", mode_label="intra")
        self.inter_points = self._load_points("multi_node_sendrecv_*.txt", mode_label="inter")

    def _load_points(self, pattern: str, mode_label: str) -> Tuple[_P2PTablePoint, ...]:
        matched = sorted(self.profile_dir.glob(pattern))
        if not matched:
            raise FileNotFoundError(
                f"collective-sim p2p measured profile missing {mode_label} file under {self.profile_dir} "
                f"(pattern={pattern})"
            )
        latest_file = matched[-1]

        points = []
        for raw_line in latest_file.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "Warming up" in line:
                continue
            matched_line = _SENDRECV_LINE_PATTERN.match(line)
            if matched_line is None:
                continue

            size_bytes = int(matched_line.group("size"))
            latency_ms = float(matched_line.group("time_us")) / 1000.0
            if size_bytes <= 0 or latency_ms <= 0.0:
                continue
            points.append(_P2PTablePoint(size_bytes=size_bytes, latency_ms=latency_ms))

        if len(points) < 2:
            raise ValueError(
                f"collective-sim p2p measured profile has insufficient samples in {latest_file}: {len(points)}"
            )

        points = sorted(points, key=lambda point: point.size_bytes)
        return tuple(points)

    @staticmethod
    def _interpolate(points: Sequence[_P2PTablePoint], size_bytes: int) -> float:
        sizes = [point.size_bytes for point in points]
        idx = bisect_left(sizes, int(size_bytes))
        if idx < len(points) and points[idx].size_bytes == size_bytes:
            return points[idx].latency_ms
        if idx <= 0:
            return points[0].latency_ms
        if idx >= len(points):
            p0 = points[-2]
            p1 = points[-1]
            delta_bytes = p1.size_bytes - p0.size_bytes
            if delta_bytes <= 0:
                return p1.latency_ms
            slope = (p1.latency_ms - p0.latency_ms) / float(delta_bytes)
            return p1.latency_ms + slope * float(size_bytes - p1.size_bytes)

        left = points[idx - 1]
        right = points[idx]
        if left.size_bytes == right.size_bytes:
            return right.latency_ms

        # Log-log interpolation provides stable extrapolation for bandwidth-like curves.
        left_size = float(left.size_bytes)
        right_size = float(right.size_bytes)
        target_size = float(size_bytes)
        left_lat = max(left.latency_ms, 1e-9)
        right_lat = max(right.latency_ms, 1e-9)
        alpha = (math.log(target_size) - math.log(left_size)) / (
            math.log(right_size) - math.log(left_size)
        )
        return math.exp(math.log(left_lat) + alpha * (math.log(right_lat) - math.log(left_lat)))

    def predict_latency_ms(self, *, size_bytes: int, is_intra_node: bool) -> float:
        points = self.intra_points if is_intra_node else self.inter_points
        return float(self._interpolate(points, int(size_bytes)))


@register_cc_backend("collective-sim")
class CollectiveSimCCBackend(CCBackend):
    backend_name = "collective-sim"

    def __init__(self, simulator_config: Any) -> None:
        super().__init__(simulator_config)
        self.options = get_backend_options(simulator_config, self.backend_name)
        self.enable_prediction_cache = bool(self.options.get("enable_prediction_cache", True))
        self.cache_path: Optional[Path] = None
        self._prediction_cache: Dict[str, float] = {}
        self._prediction_cache_dirty = False

        self.repo_root = self._resolve_repo_root()
        self.predict_collective_time = self._load_predictor(self.repo_root)

        hardware = getattr(simulator_config, "hardware", None)
        profile_name = self._resolve_profile_name(hardware, self.options)
        profile_defaults = _GPU_PROFILE_DEFAULTS.get(profile_name, {})

        self.default_gpus_per_server = int(
            self.options.get(
                "gpus_per_server",
                getattr(hardware, "gpus_per_node", 8),
            )
        )

        self.topology_kind = str(
            self.options.get("topology_kind", profile_defaults.get("topology_kind", "rail"))
        )
        self.topology_spines = int(
            self.options.get("topology_spines", profile_defaults.get("topology_spines", 8))
        )
        self.topology_servers_per_tor = int(
            self.options.get("topology_servers_per_tor", self.default_gpus_per_server)
        )
        self.topology_paths = int(
            self.options.get("topology_paths", profile_defaults.get("topology_paths", 8))
        )

        placement_order = self.options.get("placement_order", ("TP", "CP", "DP", "EP"))
        self.placement_order = tuple(str(dim).upper() for dim in placement_order)
        if not self.placement_order:
            raise ValueError("collective-sim backend requires non-empty placement_order")
        if any(dim not in _SUPPORTED_DOMAIN_DIMS for dim in self.placement_order):
            raise ValueError(
                "collective-sim backend got invalid placement_order. "
                f"Expected dims in {_SUPPORTED_DOMAIN_DIMS}, got {self.placement_order}."
            )

        default_intra = dict(profile_defaults.get("intra_server", {"model": "legacy_fabric"}))
        configured_intra = self.options.get("intra_server", {})
        self.intra_server = {**default_intra, **dict(configured_intra)}

        default_network = dict(profile_defaults.get("network", {}))
        configured_network = self.options.get("network", {})
        self.network = {**default_network, **dict(configured_network)}

        self.pp_domain_dim = self._normalize_optional_domain_dim(self.options.get("pp_domain_dim", "DP"))
        self.enforce_nonzero_duration = bool(self.options.get("enforce_nonzero_duration", True))
        self.placement_mode = str(self.options.get("placement_mode", "auto")).strip().lower()
        if self.placement_mode not in {"auto", "group_size", "global"}:
            raise ValueError(
                "collective-sim backend placement_mode must be one of: auto | group_size | global. "
                f"Got {self.placement_mode!r}."
            )
        self.strict_mpu_alignment = bool(self.options.get("strict_mpu_alignment", True))
        self.p2p_unidir_scale = float(self.options.get("p2p_unidir_scale", 1.0))
        if self.p2p_unidir_scale <= 0.0:
            raise ValueError(
                "collective-sim backend p2p_unidir_scale must be > 0 when configured."
            )

        self.p2p_measured_table: Optional[_P2PMeasuredTable] = None
        p2p_profile_dir = self.options.get("p2p_profile_dir")
        if p2p_profile_dir:
            profile_dir_path = Path(str(p2p_profile_dir))
            if not profile_dir_path.exists():
                raise FileNotFoundError(
                    f"collective-sim p2p_profile_dir not found: {profile_dir_path}"
                )
            self.p2p_measured_table = _P2PMeasuredTable(profile_dir=profile_dir_path)

        self._init_prediction_cache()

    @staticmethod
    def _to_jsonable(obj: Any) -> Any:
        if obj is None or isinstance(obj, (bool, int, float, str)):
            return obj
        if isinstance(obj, dict):
            return {str(key): CollectiveSimCCBackend._to_jsonable(value) for key, value in obj.items()}
        if isinstance(obj, (list, tuple, set)):
            return [CollectiveSimCCBackend._to_jsonable(item) for item in obj]
        return str(obj)

    @classmethod
    def _serialize_cache_key(cls, payload: Mapping[str, Any]) -> str:
        normalized = cls._to_jsonable(dict(payload))
        return json.dumps(normalized, sort_keys=True, separators=(",", ":"), ensure_ascii=True)

    def _init_prediction_cache(self) -> None:
        if not self.enable_prediction_cache:
            return

        raw_cache_path = self.options.get("cache_path")
        if raw_cache_path is None:
            return

        self.cache_path = Path(str(raw_cache_path))
        cache_parent = self.cache_path.parent
        cache_parent.mkdir(parents=True, exist_ok=True)

        if self.cache_path.exists():
            raw_text = self.cache_path.read_text(encoding="utf-8").strip()
            if raw_text:
                loaded = json.loads(raw_text)
                if not isinstance(loaded, dict):
                    raise ValueError(
                        f"collective-sim cache file must store JSON object, got {type(loaded)!r}: {self.cache_path}"
                    )
                for key, value in loaded.items():
                    self._prediction_cache[str(key)] = float(value)
        atexit.register(self._flush_prediction_cache)

    def _flush_prediction_cache(self) -> None:
        if not self.enable_prediction_cache or self.cache_path is None:
            return
        if not self._prediction_cache_dirty:
            return

        serialized = {key: float(value) for key, value in self._prediction_cache.items()}
        self.cache_path.write_text(
            json.dumps(serialized, sort_keys=True, separators=(",", ":")),
            encoding="utf-8",
        )
        self._prediction_cache_dirty = False

    def _cache_get(self, cache_key: str) -> Optional[float]:
        if not self.enable_prediction_cache:
            return None
        value = self._prediction_cache.get(cache_key)
        if value is None:
            return None
        return float(value)

    def _cache_set(self, cache_key: str, duration_ms: float) -> None:
        if not self.enable_prediction_cache:
            return
        self._prediction_cache[cache_key] = float(duration_ms)
        self._prediction_cache_dirty = True

    @staticmethod
    def _resolve_profile_name(hardware: Any, options: Mapping[str, Any]) -> str:
        explicit = options.get("gpu_profile")
        if explicit:
            return str(explicit).strip().lower()

        gpu_type = str(getattr(getattr(hardware, "gpu_type", None), "value", "")).strip().lower()
        if gpu_type == "h100-sxm":
            return "h100_sxm"
        if gpu_type == "h800-sxm":
            return "h800_sxm"
        if gpu_type == "a100-sxm":
            return "a100_sxm"
        if gpu_type == "a800-sxm":
            return "a800_sxm"
        return "h800_sxm"

    @staticmethod
    def _normalize_optional_domain_dim(raw_dim: Any) -> Optional[str]:
        if raw_dim is None:
            return None
        normalized = str(raw_dim).strip().upper()
        if normalized not in _SUPPORTED_DOMAIN_DIMS:
            raise ValueError(
                f"Invalid domain dim {raw_dim!r}. Expected one of {sorted(_SUPPORTED_DOMAIN_DIMS)}."
            )
        return normalized

    def _resolve_repo_root(self) -> Path:
        configured = self.options.get("repo_root")
        if configured:
            repo_root = Path(configured)
        else:
            repo_root = Path(__file__).resolve().parent / "collective-sim"

        if not repo_root.exists():
            raise FileNotFoundError(
                f"collective-sim repo_root not found: {repo_root}. "
                "Set communication.backend_options.repo_root to your collective-sim path. "
                "If you use submodules, run: "
                "`git submodule update --init --recursive src/core/cc_backend/collective-sim`."
            )
        return repo_root

    @staticmethod
    def _load_predictor(repo_root: Path):
        python_dir = repo_root / "python"
        if not python_dir.exists():
            raise FileNotFoundError(f"collective-sim python package directory missing: {python_dir}")

        python_dir_str = str(python_dir)
        if python_dir_str not in sys.path:
            sys.path.insert(0, python_dir_str)

        from collective_sim_core import predict_collective_time  # pylint: disable=import-error

        return predict_collective_time

    def _resolve_domain_dims(
        self,
        request: CommunicationPredictionRequest,
        collective_kind: str,
    ) -> Tuple[str, ...]:
        if request.domain_dims:
            domain_dims = tuple(str(dim).upper() for dim in request.domain_dims)
        elif collective_kind == "p2p" and str(request.group_kind or "").strip().lower() == "pp":
            if self.pp_domain_dim is None:
                raise ValueError(
                    "collective-sim backend cannot infer domain_dims for pipeline p2p traffic. "
                    "Please provide request.domain_dims explicitly or set "
                    "communication.backend_options.collective-sim.pp_domain_dim."
                )
            domain_dims = (self.pp_domain_dim,)
        else:
            domain_dims = infer_domain_dims(request.group_kind, request.op_name)

        for dim in domain_dims:
            if dim not in _SUPPORTED_DOMAIN_DIMS:
                raise ValueError(
                    f"Unsupported domain dim {dim!r}. Expected one of {sorted(_SUPPORTED_DOMAIN_DIMS)}."
                )
        return domain_dims

    def _build_parallelism(self, group_size: int, domain_dims: Tuple[str, ...]) -> Dict[str, int]:
        if len(domain_dims) != 1:
            raise ValueError(
                "collective-sim backend currently supports single-dimension domain only. "
                f"Got domain_dims={domain_dims}."
            )

        dim = domain_dims[0]
        parallelism = {"tp": 1, "cp": 1, "dp": 1, "ep": 1}

        if dim == "TP":
            parallelism["tp"] = group_size
        elif dim == "CP":
            parallelism["cp"] = group_size
        elif dim == "DP":
            parallelism["dp"] = group_size
        elif dim == "EP":
            parallelism["ep"] = group_size
        else:
            raise ValueError(f"Unsupported domain dim for collective-sim backend: {dim}")

        return parallelism

    def _build_collective_options(
        self,
        request: CommunicationPredictionRequest,
        collective_kind: str,
        domain_dims: Tuple[str, ...],
        participant_ranks: Sequence[int] = (),
    ) -> Dict[str, Any]:
        collective: Dict[str, Any] = {
            "kind": collective_kind,
            "tensor_bytes": int(request.data_size_bytes),
            "domain_dims": list(domain_dims),
            "placement_order": list(self.placement_order),
            "exclude_intra_server": bool(self.options.get("exclude_intra_server", True)),
        }

        optional_algo_keys = (
            "use_triggers",
            "allreduce_model",
            "allgather_model",
            "reducescatter_model",
            "alltoall_model",
            "alltoall_channels",
            "alltoall_chunk_bytes",
            "alltoall_chunk_inflight_per_peer",
            "nchannels",
            "intra_server_combine_rule",
        )
        for key in optional_algo_keys:
            if key in self.options:
                collective[key] = self.options[key]

        if participant_ranks:
            collective["participant_ranks"] = [int(rank) for rank in participant_ranks]

        if collective_kind == "p2p":
            metadata = dict(request.metadata or {})
            src_index = int(metadata.get("p2p_src_index", self.options.get("p2p_src_index", 0)))
            dst_index = int(metadata.get("p2p_dst_index", self.options.get("p2p_dst_index", 1)))
            direction = str(
                metadata.get("p2p_direction", self.options.get("p2p_direction", "0->1"))
            ).strip()
            if src_index < 0 or src_index >= request.group_size:
                raise ValueError(
                    f"p2p_src_index={src_index} out of range for group_size={request.group_size}"
                )
            if dst_index < 0 or dst_index >= request.group_size:
                raise ValueError(
                    f"p2p_dst_index={dst_index} out of range for group_size={request.group_size}"
                )
            if src_index == dst_index:
                raise ValueError("p2p_src_index and p2p_dst_index must be different")
            if direction not in {"0->1", "1->0", "bidir"}:
                raise ValueError("p2p_direction must be one of: 0->1 | 1->0 | bidir")
            collective["p2p_src_index"] = src_index
            collective["p2p_dst_index"] = dst_index
            collective["p2p_direction"] = direction

        return collective

    @staticmethod
    def _resolve_p2p_metadata(request: CommunicationPredictionRequest) -> Tuple[int, int, str]:
        metadata = dict(request.metadata or {})
        src_index = int(metadata.get("p2p_src_index", 0))
        dst_index = int(metadata.get("p2p_dst_index", 1))
        direction = str(metadata.get("p2p_direction", "0->1")).strip()

        if src_index < 0 or src_index >= request.group_size:
            raise ValueError(
                f"p2p_src_index={src_index} out of range for group_size={request.group_size}"
            )
        if dst_index < 0 or dst_index >= request.group_size:
            raise ValueError(
                f"p2p_dst_index={dst_index} out of range for group_size={request.group_size}"
            )
        if src_index == dst_index:
            raise ValueError("p2p_src_index and p2p_dst_index must be different")
        if direction not in {"0->1", "1->0", "bidir"}:
            raise ValueError("p2p_direction must be one of: 0->1 | 1->0 | bidir")
        return src_index, dst_index, direction

    def _predict_p2p_from_measured_table(
        self,
        request: CommunicationPredictionRequest,
    ) -> Optional[float]:
        if self.p2p_measured_table is None:
            return None

        src_index, dst_index, direction = self._resolve_p2p_metadata(request)
        src_rank = int(request.comm_group[src_index])
        dst_rank = int(request.comm_group[dst_index])
        gpus_per_server = max(1, int(self.default_gpus_per_server))
        is_intra_node = (src_rank // gpus_per_server) == (dst_rank // gpus_per_server)
        predicted_ms = self.p2p_measured_table.predict_latency_ms(
            size_bytes=int(request.data_size_bytes),
            is_intra_node=is_intra_node,
        )
        if direction in {"0->1", "1->0"}:
            predicted_ms *= self.p2p_unidir_scale
        return float(predicted_ms)

    @staticmethod
    def _normalize_group_list(raw_groups: Any) -> Tuple[Tuple[int, ...], ...]:
        if raw_groups is None:
            return ()
        if isinstance(raw_groups, (list, tuple)) and raw_groups:
            if isinstance(raw_groups[0], int):
                return (tuple(int(rank) for rank in raw_groups),)
            normalized_groups = []
            for group in raw_groups:
                if isinstance(group, (list, tuple)):
                    normalized_groups.append(tuple(int(rank) for rank in group))
            return tuple(normalized_groups)
        return ()

    def _resolve_mpu_group_candidates(
        self,
        mpu_info: Any,
        group_kind: str,
    ) -> Tuple[Tuple[int, ...], ...]:
        kind = str(group_kind or "").strip().lower()
        if kind == "tp":
            return self._normalize_group_list(getattr(mpu_info, "tp_groups", None))
        if kind == "dp":
            return self._normalize_group_list(getattr(mpu_info, "dp_groups", None))
        if kind == "cp":
            return self._normalize_group_list(getattr(mpu_info, "cp_groups", None))
        if kind == "exp":
            return self._normalize_group_list(getattr(mpu_info, "exp_groups", None))
        if kind == "ep":
            return self._normalize_group_list(getattr(mpu_info, "ep_groups", None))
        if kind == "exp_dp":
            return self._normalize_group_list(getattr(mpu_info, "dp_modulo_exp_groups", None))
        return ()

    def _resolve_global_gpus_per_server(self, world_size: int) -> int:
        preferred = max(1, int(self.default_gpus_per_server))
        if world_size <= preferred:
            return world_size
        if world_size % preferred == 0:
            return preferred
        for candidate in range(min(preferred, world_size), 0, -1):
            if world_size % candidate == 0:
                return candidate
        raise ValueError(f"Cannot resolve gpus_per_server for world_size={world_size}")

    def _build_global_placement_context(
        self,
        request: CommunicationPredictionRequest,
    ) -> Optional[Dict[str, Any]]:
        mpu_info = request.mpu_info
        if mpu_info is None:
            if self.placement_mode == "global":
                raise ValueError(
                    "collective-sim backend placement_mode='global' requires request.mpu_info"
                )
            return None
        if self.placement_mode == "group_size":
            return None

        tp_size = int(getattr(mpu_info, "tp_size", 1) or 1)
        cp_size = int(getattr(mpu_info, "cp_size", 1) or 1)
        dp_size = int(getattr(mpu_info, "dp_size", 1) or 1)
        exp_size = int(getattr(mpu_info, "exp_size", 1) or 1)
        ep_size = int(getattr(mpu_info, "ep_size", 0) or 0)

        group_kind = str(request.group_kind or "").strip().lower()
        ep_dim_size = exp_size
        if group_kind == "ep" and ep_size > 0:
            ep_dim_size = ep_size
        ep_dim_size = max(1, ep_dim_size)

        mpu_world_size = int(getattr(mpu_info, "world_size", 0) or 0)
        denominator = int(tp_size) * int(cp_size) * int(ep_dim_size)
        if denominator <= 0:
            raise ValueError(
                f"Invalid TP/CP/EP dimensions in mpu_info: tp={tp_size}, cp={cp_size}, ep={ep_dim_size}"
            )
        if mpu_world_size:
            if mpu_world_size % denominator != 0:
                raise ValueError(
                    "MPU world_size is not divisible by TP*CP*EP in placement-aware mode: "
                    f"world_size={mpu_world_size}, tp={tp_size}, cp={cp_size}, ep={ep_dim_size}"
                )
            effective_dp_size = mpu_world_size // denominator
        else:
            effective_dp_size = int(dp_size)

        parallelism = {
            "tp": tp_size,
            "cp": cp_size,
            "dp": max(1, int(effective_dp_size)),
            "ep": ep_dim_size,
        }
        world_from_parallelism = (
            int(parallelism["tp"])
            * int(parallelism["cp"])
            * int(parallelism["dp"])
            * int(parallelism["ep"])
        )
        if mpu_world_size and mpu_world_size != world_from_parallelism:
            if self.strict_mpu_alignment:
                raise ValueError(
                    "MPU world_size is inconsistent with TP/CP/DP/EP dimensions in placement-aware mode: "
                    f"world_size={mpu_world_size}, tp={parallelism['tp']}, cp={parallelism['cp']}, "
                    f"dp={parallelism['dp']}, ep={parallelism['ep']}"
                )
            world_size = world_from_parallelism
        else:
            world_size = mpu_world_size or world_from_parallelism

        participants = tuple(int(rank) for rank in request.comm_group)
        if len(participants) != len(set(participants)):
            raise ValueError(f"comm_group contains duplicate ranks: {participants}")
        if any(rank < 0 or rank >= world_size for rank in participants):
            raise ValueError(
                f"comm_group has out-of-range ranks for world_size={world_size}: {participants}"
            )

        if self.strict_mpu_alignment:
            candidate_groups = self._resolve_mpu_group_candidates(mpu_info, group_kind)
            if candidate_groups:
                participant_set = set(participants)
                if not any(participant_set == set(group) for group in candidate_groups):
                    raise ValueError(
                        "comm_group does not align with mpu_info group partition for placement-aware mode. "
                        f"group_kind={group_kind}, comm_group={participants}"
                    )

        gpus_per_server = self._resolve_global_gpus_per_server(world_size)
        if world_size % gpus_per_server != 0:
            raise ValueError(
                f"world_size={world_size} must be divisible by gpus_per_server={gpus_per_server}"
            )
        servers = world_size // gpus_per_server

        return {
            "parallelism": parallelism,
            "servers": int(servers),
            "gpus_per_server": int(gpus_per_server),
            "participant_ranks": participants,
        }

    def predict(self, request: CommunicationPredictionRequest) -> float:
        self.validate_request(request)
        if request.data_size_bytes <= 0:
            # Metadata-only communication events may have unknown tensor payload.
            # Keep deterministic behavior and avoid invalid htsim inputs.
            return 0.0

        collective_kind = infer_collective_kind(request.op_name)
        if collective_kind == "broadcast":
            raise NotImplementedError("collective-sim backend does not support broadcast yet")
        if request.group_size < 2:
            raise ValueError(
                f"collective-sim backend requires group_size >= 2 for collective={collective_kind}"
            )

        domain_dims = self._resolve_domain_dims(request, collective_kind)

        placement_context = self._build_global_placement_context(request)
        if collective_kind == "p2p":
            measured_ms = self._predict_p2p_from_measured_table(request)
            if measured_ms is not None:
                p2p_cache_key = self._serialize_cache_key(
                    {
                        "source": "p2p_measured_table",
                        "size_bytes": int(request.data_size_bytes),
                        "comm_group": list(request.comm_group),
                        "group_kind": request.group_kind,
                        "metadata": dict(request.metadata or {}),
                        "p2p_unidir_scale": float(self.p2p_unidir_scale),
                        "profile_dir": str(getattr(self.p2p_measured_table, "profile_dir", "")),
                    }
                )
                cached_p2p = self._cache_get(p2p_cache_key)
                if cached_p2p is not None:
                    return cached_p2p
                if (
                    self.enforce_nonzero_duration
                    and request.data_size_bytes > 0
                    and measured_ms <= 0.0
                ):
                    raise RuntimeError(
                        "collective-sim measured p2p table returned non-positive latency for non-zero payload. "
                        f"op={request.op_name}, group_kind={request.group_kind}, "
                        f"group_size={request.group_size}, data_size_bytes={request.data_size_bytes}."
                    )
                self._cache_set(p2p_cache_key, measured_ms)
                return measured_ms

        if placement_context is None:
            parallelism = self._build_parallelism(request.group_size, domain_dims)
            gpus_per_server = max(1, min(self.default_gpus_per_server, request.group_size))
            if request.group_size % gpus_per_server == 0:
                servers = request.group_size // gpus_per_server
            else:
                servers = 1
                gpus_per_server = request.group_size
            participant_ranks: Tuple[int, ...] = ()
        else:
            parallelism = placement_context["parallelism"]
            servers = int(placement_context["servers"])
            gpus_per_server = int(placement_context["gpus_per_server"])
            participant_ranks = tuple(placement_context["participant_ranks"])

        scenario = {
            "cluster": {
                "servers": servers,
                "gpus_per_server": gpus_per_server,
            },
            "parallelism": parallelism,
            "topology": {
                "kind": self.topology_kind,
                "spines": self.topology_spines,
                "servers_per_tor": self.topology_servers_per_tor,
                "paths": self.topology_paths,
            },
            "collective": self._build_collective_options(
                request,
                collective_kind,
                domain_dims,
                participant_ranks=participant_ranks,
            ),
            "intra_server": self.intra_server,
        }

        if self.network:
            scenario["network"] = self.network

        scenario_cache_key = self._serialize_cache_key(
            {
                "source": "collective_sim",
                "scenario": scenario,
                "repo_root": str(self.repo_root),
            }
        )
        cached_duration = self._cache_get(scenario_cache_key)
        if cached_duration is not None:
            return cached_duration

        result = self.predict_collective_time(scenario, repo_root=self.repo_root)
        if "predicted_time_ms" not in result:
            raise RuntimeError(
                "collective-sim backend received invalid result without predicted_time_ms"
            )

        predicted_time_ms = float(result["predicted_time_ms"])
        if self.enforce_nonzero_duration and request.data_size_bytes > 0 and predicted_time_ms <= 0.0:
            raise RuntimeError(
                "collective-sim returned non-positive predicted_time_ms for non-zero payload. "
                f"op={request.op_name}, group_kind={request.group_kind}, domain_dims={domain_dims}, "
                f"group_size={request.group_size}, data_size_bytes={request.data_size_bytes}. "
                "Please verify collective semantics and cluster/intra-server settings."
            )
        self._cache_set(scenario_cache_key, predicted_time_ms)
        return predicted_time_ms
