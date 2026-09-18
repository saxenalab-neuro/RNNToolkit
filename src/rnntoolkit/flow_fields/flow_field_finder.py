import torch
import torch.nn as nn
from typing import Tuple
from rnntoolkit.linear import Linearization
from rnntoolkit.flow_fields.flow_field import FlowField
from rnntoolkit.flow_fields.flow_field_finder_base import FlowFieldFinderBase
from rnntoolkit.adapter import RNNAdapter


class FlowFieldFinder(FlowFieldFinderBase):
    def __init__(
        self,
        rnn: nn.RNN | nn.GRU | nn.LSTM,
        num_points: int = 10,
        x_offset: int = 1,
        y_offset: int = 1,
        x_center: float = 0.0,
        y_center: float = 0.0,
        fit_states: torch.Tensor | tuple[torch.Tensor, torch.Tensor] | None = None,
        axes: torch.Tensor | None = None,
        follow_traj: bool = False,
    ):
        """Configure two-dimensional flow analysis for an RNN, GRU, or LSTM.

        State width D is H for RNN/GRU and 2H for LSTM. PCA and explicit
        axes operate on the complete state, including the LSTM cell state.

        Args:
            rnn: Single-layer, unidirectional nn.RNN, nn.GRU, or nn.LSTM
                without projections. Either batch_first setting is accepted.
            num_points: Grid samples per axis; each field has num_points squared points.
            x_offset: Grid half-width along the first reduced axis.
            y_offset: Grid half-width along the second reduced axis.
            x_center: Fixed grid center on the first reduced axis.
            y_center: Fixed grid center on the second reduced axis.
            fit_states: Optional PCA fitting data [..., D], or an LSTM tuple
                with matching [..., H] components. Without fitting data,
                the first flow call fits PCA on its states. PCA requires
                at least two samples and two state features.
            axes: Optional projection matrix [2, D]; overrides PCA for
                projection and uses a pseudoinverse to reconstruct states.
            follow_traj: Center each grid on its projected state when True;
                otherwise use x_center and y_center.

        Note:
            The PCA path uses scikit-learn; supply detached CPU states.
            Custom axes for LSTMs must have width 2H.
        """
        self.adapter = RNNAdapter(rnn)
        if fit_states is not None and not self.adapter.is_packed(fit_states):
            fit_states = self.adapter.pack_state(fit_states)
        super().__init__(
            rnn=rnn,
            num_points=num_points,
            x_offset=x_offset,
            y_offset=y_offset,
            x_center=x_center,
            y_center=y_center,
            fit_states=fit_states,
            axes=axes,
        )

        self.follow_traj = follow_traj

        # Need to define a valid linearization object in each child of FlowFieldFinderBase
        self.linearization = Linearization(self.rnn)

    def find_nonlinear_flow(
        self,
        states: torch.Tensor | Tuple,
        inp: torch.Tensor,
    ) -> list:
        """Compute projected one-step displacements on a grid for each sample.

        Args:
            states: State tensor [D] or [..., D]. LSTMs also accept (h, c)
                tuples with matching [..., H] components. Packed LSTM
                tensors use [h, c] order. Leading dimensions are flattened.
            inp: Inputs [I] or [..., I], with one input per flattened state.
                A single input is not automatically repeated across states.

        Returns:
            List of FlowField objects in flattened sample order. Each contains
            grid coordinates [P, P, 2], displacements x_vels/y_vels [P, P],
            and speeds [P, P] normalized by the largest speed in that field,
            where P is num_points. These are projected displacements, not
            the full-state speed or an energy function.
        """

        flow_field_list = []

        # pack if lstm is used
        if not self.adapter.is_packed(states):
            states = self.adapter.pack_state(states)
        # Reshape to nxd
        states, inp = self._nxd(states), self._nxd(inp)

        # assert states and input match shape
        assert states.shape[0] == inp.shape[0]
        n_states = states.shape[0]

        reduced_traj = self._reduce_traj(states)

        # Now going through trajectory
        for n in range(n_states):
            reduced_traj_n, inp_n = reduced_traj[n], inp[n]
            # If follow trajectory is true get grid centered around current t
            # This will make a different grid for each state (n grids)
            if self.follow_traj:
                lower_bound_x, upper_bound_x, lower_bound_y, upper_bound_y = (
                    self._set_tv_bounds(reduced_traj_n)
                )
            else:
                lower_bound_x, upper_bound_x, lower_bound_y, upper_bound_y = (
                    self._set_bounds()
                )

            low_dim_grid, inverse_grid = self._inverse_grid(
                lower_bound_x,
                upper_bound_x,
                lower_bound_y,
                upper_bound_y,
            )

            # Repeat along the batch dimension to match the grid
            full_inp_batch = inp_n.repeat(low_dim_grid.shape[0], 1)

            with torch.no_grad():
                # Current timestep input
                # Get activity for current timestep
                h = self.adapter(
                    full_inp_batch,
                    inverse_grid,
                )

            # Reduce h_next
            h_next = self._reduce_traj(h)

            # Compute velocity and speed
            x_vel, y_vel = self._compute_velocity(h_next, low_dim_grid)
            speed = self._compute_speed(x_vel, y_vel)

            # Reshape to match FlowField object requirements
            x_vel, y_vel, low_dim_grid, speed = self._reshape_vals(
                x_vel, y_vel, low_dim_grid, speed
            )
            flow_field = FlowField(x_vel, y_vel, low_dim_grid, speed)
            flow_field_list.append(flow_field)

        return flow_field_list

    def find_linear_flow(
        self,
        states: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        inp: torch.Tensor,
        delta_inp: torch.Tensor,
    ) -> list:
        """Compute flow fields using a local affine approximation at each state.

        Args:
            states: Reference states [D] or [..., D], or LSTM (h, c) tuples
                with matching [..., H] components. Packed states use [h, c].
            inp: Reference inputs [I] or [..., I], one per flattened state.
            delta_inp: Input perturbations with the same shape as inp.

        Returns:
            List of FlowField objects, one per flattened reference state,
            with the same shapes and speed normalization as nonlinear flow.
            Linearization advances the entire LSTM state before projection.

        Note:
            A stable-looking two-dimensional projection does not establish
            full-state stability. Inspect the full Jacobian eigenvalues.
        """

        # pack if lstm is used
        if not self.adapter.is_packed(states):
            states = self.adapter.pack_state(states)
        # reshape to nxd
        states, inp, delta_inp = self._nxd(states), self._nxd(inp), self._nxd(delta_inp)

        assert states.shape[0] == delta_inp.shape[0]
        assert delta_inp.shape == inp.shape
        n_states = states.shape[0]

        # Lists for x and y velocities
        flow_field_list = []

        # Reduce the regional trajectories and return pca object
        reduced_traj = self._reduce_traj(states)

        for n in range(n_states):
            states_n = states[n]
            reduced_traj_n = reduced_traj[n]
            inp_n = inp[n]
            delta_inp_n = delta_inp[n]

            # If follow trajectory is true get grid centered around current t
            # This will make a different grid for each state (n grids)
            if self.follow_traj:
                lower_bound_x, upper_bound_x, lower_bound_y, upper_bound_y = (
                    self._set_tv_bounds(reduced_traj_n)
                )
            else:
                lower_bound_x, upper_bound_x, lower_bound_y, upper_bound_y = (
                    self._set_bounds()
                )

            # Inverse the grid to pass through RNN
            low_dim_grid, inverse_grid = self._inverse_grid(
                lower_bound_x,
                upper_bound_x,
                lower_bound_y,
                upper_bound_y,
            )

            # Get a perturbation of the activity
            delta_h = inverse_grid - states_n

            with torch.no_grad():
                # call forward method for linearization to get affine transformation
                h = self.linearization(inp_n, states_n, delta_inp_n, delta_h)

            # Put next h into a grid format
            h_next = self._reduce_traj(h)

            # Compute velocities between gathered trajectory of grid and original grid values
            x_vel, y_vel = self._compute_velocity(h_next, low_dim_grid)
            speed = self._compute_speed(x_vel, y_vel)

            x_vel, y_vel, low_dim_grid, speed = self._reshape_vals(
                x_vel, y_vel, low_dim_grid, speed
            )

            # Reshape data back to grid
            flow_field = FlowField(x_vel, y_vel, low_dim_grid, speed)
            flow_field_list.append(flow_field)

        return flow_field_list
