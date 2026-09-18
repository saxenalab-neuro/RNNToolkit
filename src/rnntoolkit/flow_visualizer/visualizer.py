from collections.abc import Sequence

import pygame
import torch
import torch.nn as nn
from rnntoolkit.flow_fields.flow_field import FlowField
from rnntoolkit.flow_fields.flow_field_finder import FlowFieldFinder
from rnntoolkit.flow_fields.flow_field_finder_base import FlowFieldFinderBase
from rnntoolkit.flow_visualizer.axes_panel import AxesPreferencesPanel
from rnntoolkit.flow_visualizer.axis_selection import AxisSelectionMixin
from rnntoolkit.flow_visualizer.visualizer_base import FlowFieldVisualizerBase

pygame.init()


class FlowFieldVisualizer(AxisSelectionMixin, FlowFieldVisualizerBase):
    """Interactive two-dimensional viewer for RNN, GRU, or LSTM dynamics.

    Supply packed state tensors to visualize: width H for RNN/GRU and
    2H for LSTM in [h, c] order. The viewer does not pack trajectory tuples.
    """

    def __init__(
        self,
        rnn: nn.RNN | nn.GRU | nn.LSTM,
        num_points: int = 10,
        x_offset: int = 5,
        y_offset: int = 5,
        x_center: float = 0.0,
        y_center: float = 0.0,
        fit_states: torch.Tensor | None = None,
        axes: torch.Tensor | None = None,
        flow_type: str = "nonlinear",
        axis_labels: Sequence[str] | None = None,
    ) -> None:
        """Initialize the concrete RNNToolkit flow-field visualizer.

        Args:
            rnn: Single-layer, unidirectional RNN/GRU/LSTM without projections.
            num_points: Number of grid samples along each displayed axis.
            x_offset: Initial half-width of the x-axis data window.
            y_offset: Initial half-width of the y-axis data window.
            x_center: Initial x-axis center in reduced coordinates.
            y_center: Initial y-axis center in reduced coordinates.
            fit_states: Packed states [..., D] used to fit PCA when axes is omitted.
                D is H for RNN/GRU and 2H for LSTM in [h, c] order.
            axes: Optional axis bank with shape ``[N, D]``, where
                ``N >= 2``. The first two axes are selected initially.
            flow_type: Either ``"nonlinear"`` or ``"linear"``.
            axis_labels: Optional display names for the axes. Its length must
                equal ``N``. Defaults to ``axis_1``, ``axis_2``, and so on.
        """
        selected_axes = self._initialize_axes(axes, axis_labels)
        super().__init__(
            rnn,
            num_points,
            x_offset,
            y_offset,
            x_center,
            y_center,
            fit_states,
            selected_axes,
            flow_type,
        )
        self.preferences_panel = AxesPreferencesPanel(self.pref_btn, self)

    def build_finder(self) -> FlowFieldFinder:
        """Build a finder using the currently selected two-axis basis."""
        return FlowFieldFinder(
            rnn=self.rnn,
            num_points=self.num_points,
            x_offset=self.x_offset,
            y_offset=self.y_offset,
            x_center=self.x_center,
            y_center=self.y_center,
            fit_states=self.fit_states,
            axes=self.axes,
            follow_traj=False,
        )

    def prepare_data(
        self,
        inputs: torch.Tensor,
        states: torch.Tensor,
        delta_inputs: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Flatten inputs and packed trajectories into one sample per page.

        Args:
            inputs: Input tensor [I] or [..., I].
            states: Packed state tensor [D] or [..., D], with D = 2H for
                LSTM. Pack (h, c) tuples before passing them to the viewer.
            delta_inputs: Optional perturbations aligned with inputs.

        Returns:
            Inputs [N, I], states [N, D], and perturbations [N, I] or None.
            Inputs and states must have matching sample counts.
        """
        delta_inp_nxd = (
            FlowFieldFinderBase._nxd(delta_inputs)
            if delta_inputs is not None
            else None
        )
        return (
            FlowFieldFinderBase._nxd(inputs),
            FlowFieldFinderBase._nxd(states),
            delta_inp_nxd,
        )

    def compute_flow_field(
        self,
        inp_nxd: torch.Tensor,
        states_nxd: torch.Tensor,
        delta_inp_nxd: torch.Tensor | None = None,
    ) -> FlowField:
        """Compute and cache the current page's projected flow field.

        Args:
            inp_nxd: Reference inputs [N, I].
            states_nxd: Packed states [N, D], including [h, c] for LSTMs.
            delta_inp_nxd: Optional input perturbations [N, I]; linear mode
                uses zero perturbations when omitted.

        Returns:
            FlowField for the current page and selected two-axis projection.
        """
        state_n = states_nxd[self.current_element_idx]
        inp_n = inp_nxd[self.current_element_idx]
        delta_inp_n = (
            delta_inp_nxd[self.current_element_idx]
            if delta_inp_nxd is not None
            else None
        )

        finder = self.current_field()
        finder.num_points = self.preferences["grid_points"]
        finder.x_offset = self.view_span / 2.0
        finder.y_offset = self.view_span / 2.0
        finder.x_center = (self.x_bounds[0] + self.x_bounds[1]) / 2.0
        finder.y_center = (self.y_bounds[0] + self.y_bounds[1]) / 2.0

        if state_n.dim() == 1:
            state_n = state_n.unsqueeze(0)
        if inp_n.dim() == 1:
            inp_n = inp_n.unsqueeze(0)

        with torch.no_grad():
            if self.flow_type == "linear":
                if delta_inp_n is None:
                    delta_inp_n = torch.zeros_like(inp_n)
                if delta_inp_n.dim() == 1:
                    delta_inp_n = delta_inp_n.unsqueeze(0)
                flow = finder.find_linear_flow(state_n, inp_n, delta_inp_n)[0]
            else:
                flow = finder.find_nonlinear_flow(state_n, inp_n)[0]

        self._flow_cache = {
            "grid": flow.grid.detach().cpu().numpy(),
            "x_vel": flow.x_vels.detach().cpu().numpy(),
            "y_vel": flow.y_vels.detach().cpu().numpy(),
            "speed": flow.speeds.detach().cpu().numpy(),
        }
        self._flow_dirty = False
        return flow
