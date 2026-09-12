"""Reusable explicit-axis selection behavior for flow-field visualizers."""

from collections.abc import Sequence
from typing import Any

import torch


class AxisSelectionMixin:
    """Manage a selectable bank of axes for a flow-field visualizer."""

    all_axes: torch.Tensor | None
    axis_labels: list[str]
    selected_axis_indices: list[int]
    axes: torch.Tensor | None
    pages: list[Any]
    current_page: int
    _flow_cache: Any

    def _initialize_axes(
        self,
        axes: torch.Tensor | None,
        axis_labels: Sequence[str] | None,
    ) -> torch.Tensor | None:
        """Validate an axis bank and return its initial two-axis selection."""
        if axes is None:
            if axis_labels is not None:
                raise ValueError("axis_labels requires axes")
            self.all_axes = None
            self.axis_labels = []
            self.selected_axis_indices = []
            return None

        if axes.dim() != 2 or axes.shape[0] < 2:
            raise ValueError("axes must have shape [N, D] with N >= 2")

        n_axes = axes.shape[0]
        if axis_labels is None:
            labels = [f"axis_{index + 1}" for index in range(n_axes)]
        else:
            labels = list(axis_labels)
            if len(labels) != n_axes:
                raise ValueError("axis_labels must have the same length as axes")
            if not all(isinstance(label, str) for label in labels):
                raise TypeError("every axis label must be a string")

        self.all_axes = axes
        self.axis_labels = labels
        self.selected_axis_indices = [0, 1]
        return axes[self.selected_axis_indices]

    def toggle_axis(self, index: int) -> None:
        """Stage an axis toggle and rebuild only for a valid two-axis set.

        Any number of toggles may be enabled while the user edits the
        selection. The currently displayed flow remains unchanged until the
        staged selection contains exactly two axes.
        """
        if self.all_axes is None or not 0 <= index < len(self.axis_labels):
            return

        selected = set(self.selected_axis_indices)
        if index in selected:
            selected.remove(index)
        else:
            selected.add(index)
        self.selected_axis_indices = [
            axis_index
            for axis_index in range(len(self.axis_labels))
            if axis_index in selected
        ]

        if len(self.selected_axis_indices) != 2:
            return

        self.axes = self.all_axes[self.selected_axis_indices]
        self.pages = [self.build_finder()]
        self.current_page = 1
        self._flow_cache = None
        self._mark_dirty()
