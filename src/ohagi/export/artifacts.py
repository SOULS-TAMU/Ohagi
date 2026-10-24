"""Artifact-directory names and explicit path preparation."""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ArtifactPaths:
    """Filesystem locations associated with a saved model bundle."""

    root: Path
    model: Path
    metadata: Path
    weights: Path
    problem_arrays: Path
    projection_arrays: Path
    normalization: Path
    history: Path
    metrics: Path


def artifact_paths(root: str | Path, *, create: bool = False) -> ArtifactPaths:
    """Resolve the standard artifact locations relative to a metadata file."""
    root_path = Path(root)
    if create:
        root_path.mkdir(parents=True, exist_ok=True)
    return ArtifactPaths(
        root=root_path,
        model=root_path / "metadata.json",
        metadata=root_path / "metadata.json",
        weights=root_path / "weights.npz",
        problem_arrays=root_path / "problem_arrays.npz",
        projection_arrays=root_path / "projection_arrays.npz",
        normalization=root_path / "normalization.npz",
        history=root_path / "history.csv",
        metrics=root_path / "metrics.json",
    )
