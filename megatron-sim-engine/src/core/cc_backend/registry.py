"""Backend registry and factory for communication predictors."""

from __future__ import annotations

from typing import Any, Callable, Dict, Iterable

from .base import CCBackend


BackendFactory = Callable[[Any], CCBackend]
_BACKEND_FACTORIES: Dict[str, BackendFactory] = {}
_BUILTINS_LOADED = False


def register_cc_backend(name: str) -> Callable[[BackendFactory], BackendFactory]:
    """Register a backend factory under a stable backend name."""

    normalized = name.strip().lower()
    if not normalized:
        raise ValueError("backend name must not be empty")

    def _decorator(factory: BackendFactory) -> BackendFactory:
        if normalized in _BACKEND_FACTORIES:
            raise ValueError(f"CC backend {normalized} is already registered")
        _BACKEND_FACTORIES[normalized] = factory
        return factory

    return _decorator


def _ensure_builtin_backends_loaded() -> None:
    global _BUILTINS_LOADED
    if _BUILTINS_LOADED:
        return

    # Imported for registration side effects.
    from . import analytical_backend  # noqa: F401
    from . import collective_sim_backend  # noqa: F401
    from . import profiling_backend  # noqa: F401
    from . import cc_estimator_backend  # noqa: F401

    _BUILTINS_LOADED = True


def list_cc_backends() -> Iterable[str]:
    _ensure_builtin_backends_loaded()
    return tuple(sorted(_BACKEND_FACTORIES.keys()))


def create_cc_backend(backend_name: str, simulator_config: Any) -> CCBackend:
    """Instantiate a communication backend by name."""

    if not backend_name:
        raise ValueError("backend_name must be provided")

    normalized = backend_name.strip().lower()
    _ensure_builtin_backends_loaded()

    factory = _BACKEND_FACTORIES.get(normalized)
    if factory is None:
        available = ", ".join(sorted(_BACKEND_FACTORIES))
        raise ValueError(f"Unknown CC backend {normalized}. Available backends: {available}")

    backend = factory(simulator_config)
    if not isinstance(backend, CCBackend):
        raise TypeError(f"Backend factory {normalized} must return CCBackend, got {type(backend)!r}")
    return backend
