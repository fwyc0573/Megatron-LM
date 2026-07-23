"""Compatibility namespace for legacy `cc_backend` imports."""

from pathlib import Path
from pkgutil import extend_path

from src.core.cc_backend import *  # noqa: F401,F403

__path__ = extend_path(__path__, __name__)
_target = Path(__file__).resolve().parents[1] / "src" / "core" / "cc_backend"
if _target.exists():
    __path__.append(str(_target))
