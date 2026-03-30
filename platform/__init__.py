"""
MedAI Platform package.

This repository uses the top-level package name ``platform``, which would
normally shadow Python's standard-library ``platform`` module when the project
root is on ``sys.path``. That breaks pytest and any dependency that imports
``platform`` expecting the stdlib API.

To keep the existing package layout without breaking third-party imports, this
package re-exports the stdlib ``platform`` module's public API while still
behaving like a package for local imports such as ``platform.config.settings``.
"""

from __future__ import annotations

import importlib.util
import sysconfig
from pathlib import Path

__version__ = "1.0.0"
__app_name__ = "MedAI Platform"


def _load_stdlib_platform():
    stdlib_platform_path = Path(sysconfig.get_path("stdlib")) / "platform.py"
    spec = importlib.util.spec_from_file_location("_stdlib_platform", stdlib_platform_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load stdlib platform module from {stdlib_platform_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_STDLIB_PLATFORM = _load_stdlib_platform()

# Re-export the stdlib platform API so libraries like pytest/attrs continue to
# work even though this package name shadows the stdlib module.
for _name in dir(_STDLIB_PLATFORM):
    if _name.startswith("__"):
        continue
    globals().setdefault(_name, getattr(_STDLIB_PLATFORM, _name))


def __getattr__(name: str):
    return getattr(_STDLIB_PLATFORM, name)


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(dir(_STDLIB_PLATFORM)))
