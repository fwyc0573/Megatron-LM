"""Compatibility namespace for legacy `comm_sim` imports."""

from pathlib import Path
from pkgutil import extend_path

__path__ = extend_path(__path__, __name__)
_target = Path(__file__).resolve().parents[1] / "src" / "core" / "comm_sim"
if _target.exists():
    __path__.append(str(_target))
