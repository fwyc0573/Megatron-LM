"""Base interfaces for communication-cost backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict

from .types import CommunicationPredictionRequest


class CCBackend(ABC):
    """Abstract communication prediction backend."""

    backend_name = "base"

    def __init__(self, simulator_config: Any) -> None:
        self.simulator_config = simulator_config

    @abstractmethod
    def predict(self, request: CommunicationPredictionRequest) -> float:
        """Predict communication duration in milliseconds."""

    def validate_request(self, request: CommunicationPredictionRequest) -> None:
        if request.group_size <= 0:
            raise ValueError("request.comm_group must not be empty")
        if request.data_size_bytes < 0:
            raise ValueError("request.data_size_bytes must be >= 0")


def get_backend_options(simulator_config: Any, backend_name: str) -> Dict[str, Any]:
    """Resolve backend-specific options from simulator configuration.

    Supports both:
      1) `communication.backend_options = {"<backend>": {...}}`
      2) `communication.backend_options = {...}` for single-backend configs.
    """

    communication = getattr(simulator_config, "communication", None)
    if communication is None:
        return {}

    backend_options = getattr(communication, "backend_options", None)
    if not isinstance(backend_options, dict):
        return {}

    nested = backend_options.get(backend_name)
    if isinstance(nested, dict):
        return dict(nested)

    return dict(backend_options)
