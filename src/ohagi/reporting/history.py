"""Host-side reporting; plotting remains an optional dependency."""

import csv
from pathlib import Path


def save_history(path, history):
    """Write epoch records to the configured history artifact."""
    rows = list(history)
    if not rows:
        raise ValueError("history cannot be empty")
    keys = list(dict.fromkeys(key for row in rows for key in row))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    return path


def plot_history(history, *, keys=("loss",), ax=None):
    """Plot selected training-history columns and save the resulting figure."""
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots()
    rows = list(history)
    for key in keys:
        ax.plot([r["epoch"] for r in rows], [r[key] for r in rows], label=key)
    ax.set_xlabel("Epoch")
    ax.legend()
    return ax
