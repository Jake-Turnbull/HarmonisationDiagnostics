"""Runtime version resolution for DiagnoseHarmonisation.

When the package is imported from a Git checkout, the version is derived from
the nearest tag plus commit metadata. Installed wheels/sdists fall back to the
distribution metadata, and a baked-in version is used as a final fallback.
"""

from __future__ import annotations

from importlib import metadata as importlib_metadata
from pathlib import Path
import re
import subprocess
from typing import Union

__all__ = [
    "__version__",
    "__version_tuple__",
    "version",
    "version_tuple",
    "__commit_id__",
    "commit_id",
]

VERSION_TUPLE = tuple[Union[int, str], ...]
COMMIT_ID = Union[str, None]

_DIST_NAMES = ("HarmonizationDiagnostics",)
_FALLBACK_VERSION = "0.1.1"
_REPO_ROOT = Path(__file__).resolve().parents[1]


def _run_git_command(*args: str) -> str | None:
    """Return stripped Git command output for this repository."""
    git_dir = _REPO_ROOT / ".git"
    if not git_dir.exists():
        return None

    try:
        completed = subprocess.run(
            ["git", *args],
            cwd=_REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
            timeout=2,
        )
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None

    return completed.stdout.strip() or None


def _version_tuple_from_string(value: str) -> VERSION_TUPLE:
    tokens = re.split(r"[.+-]", value)
    version_parts: list[Union[int, str]] = []
    for token in tokens:
        if not token:
            continue
        version_parts.append(int(token) if token.isdigit() else token)
    return tuple(version_parts)


def _resolve_installed_version() -> str | None:
    for dist_name in _DIST_NAMES:
        try:
            return importlib_metadata.version(dist_name)
        except importlib_metadata.PackageNotFoundError:
            continue
    return None


def _resolve_git_version() -> tuple[str | None, COMMIT_ID]:
    describe = _run_git_command("describe", "--tags", "--long", "--dirty", "--always")
    commit = _run_git_command("rev-parse", "--short=8", "HEAD")
    if not describe:
        return None, commit

    match = re.fullmatch(
        r"(?P<tag>.+)-(?P<count>\d+)-g(?P<sha>[0-9a-f]+)(?P<dirty>-dirty)?",
        describe,
    )
    if not match:
        return describe.lstrip("v"), commit

    tag = match.group("tag").lstrip("v")
    commit_count = int(match.group("count"))
    sha = match.group("sha")
    dirty = match.group("dirty")

    if commit_count == 0 and not dirty:
        return tag, commit

    local_parts = [f"g{sha}"]
    if dirty:
        local_parts.append("dirty")
    return f"{tag}.post{commit_count}+{'.'.join(local_parts)}", commit


def _resolve_version_info() -> tuple[str, VERSION_TUPLE, COMMIT_ID]:
    git_version, git_commit = _resolve_git_version()
    if git_version:
        return git_version, _version_tuple_from_string(git_version), git_commit

    installed_version = _resolve_installed_version()
    if installed_version:
        return installed_version, _version_tuple_from_string(installed_version), None

    return _FALLBACK_VERSION, _version_tuple_from_string(_FALLBACK_VERSION), None


__version__, __version_tuple__, __commit_id__ = _resolve_version_info()
version = __version__
version_tuple = __version_tuple__
commit_id = __commit_id__
