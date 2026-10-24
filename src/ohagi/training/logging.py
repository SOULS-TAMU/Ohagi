"""Fixed-width training logs and completion banners."""

from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime

_SEPARATOR_WIDTH = 120


def print_postprocessing_start() -> None:
    """Print the KKT-HardNet-style postprocessing message."""

    print("Postprocessing...")


def print_research_citation(*, citation="", contact="") -> None:
    """Print the framework citation supplied to the training logger."""
    print("=" * _SEPARATOR_WIDTH)
    print("If you use this model in your research, please cite:")
    print(citation)
    print()
    print(f"Contact: {contact}")
    print("=" * _SEPARATOR_WIDTH)


def print_postprocessing_finish(*, include_citation=True) -> None:
    """Print the completed postprocessing status and artifact location."""
    print("Done.")
    if include_citation:
        print_research_citation()


@dataclass
class KKTTableLogger:
    """Format host-side training progress as a compact table."""

    epochs: int
    print_every: int = 1
    width: int = 15
    title: str = "Training"
    _started: bool = False
    _has_validation: bool = False

    def start(
        self,
        *,
        dims: dict,
        task: str,
        mode: str,
        samples: int,
        batch_size: int,
        network: str,
        eta,
        pretrain_epochs: int,
        consistency_weight: float,
        validation_samples: int = 0,
        dual_pretrain_gamma: float | None = None,
        predicted_dual_projection: bool = False,
    ):
        """Print the training header and initial progress information."""
        if not self.print_every:
            return
        self._has_validation = int(validation_samples) > 0
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"Model training started! [{now}]\n")
        print(self.title)
        print(
            "  dims: "
            f"n_x={dims.get('n_x')} n_y={dims.get('n_y')} "
            f"n_eq={dims.get('n_eq')} n_ineq={dims.get('n_ineq')}"
        )
        print(f"  task: {task}")
        print(f"  mode: {mode}")
        print(
            f"  samples: train={samples} val={validation_samples} batch_size={batch_size}"
        )
        print(f"  network: {network}")
        print(f"  eta: {'n/a' if eta is None else eta}")
        print(f"  epoch_mlp: {pretrain_epochs if pretrain_epochs else 'n/a'}")
        print(f"  cons_alpha: {consistency_weight}")
        if dual_pretrain_gamma is not None:
            print(
                "  dual pretrain box: "
                f"min(-min(0,-z), gamma), gamma={float(dual_pretrain_gamma):.4e}"
            )
        if predicted_dual_projection:
            print(
                "  projection dual initialization: predicted from first projected epoch"
            )
        print("=" * _SEPARATOR_WIDTH)
        group_width = self.width * 2 + 1
        print(
            f"{'Epoch':>{self.width}} | {'Training':^{group_width}} | {'Validation':^{group_width}}"
        )
        print("-" * _SEPARATOR_WIDTH)
        print(
            f"{'':>{self.width}} | "
            f"{'Loss':>{self.width}} {'Violation':>{self.width}} | "
            f"{'Loss':>{self.width}} {'Violation':>{self.width}}"
        )
        print("=" * _SEPARATOR_WIDTH)
        self._started = True

    def should_print(self, epoch: int) -> bool:
        """Decide whether the current epoch should be reported."""
        return bool(self.print_every) and (
            epoch == 1
            or epoch == self.epochs
            or epoch % max(1, int(self.print_every)) == 0
        )

    def epoch(self, epoch: int, train: dict, validation: dict | None = None):
        """Format one epoch of training and validation metrics."""
        if not self.should_print(epoch):
            return
        validation = validation or {}

        def metric(row, name, default=0.0):
            value = row.get(name, default)
            return f"{float(value):>{self.width}.4e}"

        def unavailable():
            return f"{'n/a':>{self.width}}"

        if self._has_validation and validation:
            validation_columns = (
                f"{metric(validation, 'loss')} {metric(validation, 'violation')}"
            )
        else:
            validation_columns = f"{unavailable()} {unavailable()}"

        label = f"{epoch:03d}/{self.epochs:03d}"
        print(
            f"{label:>{self.width}} | "
            f"{metric(train, 'loss')} "
            f"{metric(train, 'violation')} | "
            f"{validation_columns}"
        )

    def phase_switch(self, epoch: int, reason: str):
        """Report the transition between training phases."""
        if self.print_every:
            print(
                f"Switching to projection phase at epoch {epoch} (trigger: {reason})."
            )

    def finish(self, training_seconds: float | None = None):
        """Print the final training status and elapsed time."""
        if self.print_every and self._started:
            print("=" * _SEPARATOR_WIDTH)
            if training_seconds is not None:
                print(
                    f"Model training time (wall time) = {float(training_seconds):.4f} seconds"
                )
            print("Model training finished!")
