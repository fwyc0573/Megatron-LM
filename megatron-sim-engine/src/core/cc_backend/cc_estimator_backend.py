"""CC-estimator backend wrapper."""

from __future__ import annotations

from typing import Any

from src.extensions.cc_estimator_integration import CCEstimatorWrapper

from .base import CCBackend
from .registry import register_cc_backend
from .types import CommunicationPredictionRequest


@register_cc_backend("cc-estimator")
class CCEstimatorBackend(CCBackend):
    backend_name = "cc-estimator"

    def __init__(self, simulator_config: Any) -> None:
        super().__init__(simulator_config)
        self.estimator = CCEstimatorWrapper(simulator_config)
        if self.estimator.predictor is None:
            raise RuntimeError(
                "cc-estimator backend selected but nccl_predictor is unavailable. "
                "Install CC-estimator dependencies or switch to another backend."
            )

    def predict(self, request: CommunicationPredictionRequest) -> float:
        self.validate_request(request)
        return float(
            self.estimator.predict_communication_time(
                comm_group=list(request.comm_group),
                data_size=int(request.data_size_bytes),
                comm_func=request.op_name,
            )
        )
