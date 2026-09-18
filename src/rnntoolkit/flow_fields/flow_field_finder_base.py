import torch
import torch.nn as nn
from sklearn.decomposition import PCA
from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_is_fitted
from typing import Generic, TypeVar, Tuple

RNN = TypeVar("RNN", bound=nn.Module)


class FlowFieldFinderBase(Generic[RNN]):
    def __init__(
        self,
        rnn: RNN,
        num_points: int,
        x_offset: int,
        y_offset: int,
        x_center: float = 0.0,
        y_center: float = 0.0,
        fit_states: torch.Tensor | None = None,
        axes: torch.Tensor | None = None,
        **kwargs,
    ):
        """Initialize projection and grid helpers for a concrete flow finder.

        This base class accepts packed tensors only; subclasses handle native
        state structures and implement the recurrent update.

        Args:
            rnn: Recurrent module with parameters and a batch_first attribute.
            num_points: Samples per grid axis.
            x_offset: Half-width on the first reduced axis.
            y_offset: Half-width on the second reduced axis.
            x_center: Center on the first reduced axis.
            y_center: Center on the second reduced axis.
            fit_states: Optional packed PCA data [..., D], where D includes
                both h and c for LSTMs. Use detached CPU tensors for PCA.
            axes: Optional projection matrix [2, D], used instead of PCA.
            **kwargs: Reserved for subclass configuration.
        """
        self.rnn = rnn
        self.fit_states = fit_states
        self.num_points = num_points
        self.x_offset = x_offset
        self.y_offset = y_offset
        self.x_center = x_center
        self.y_center = y_center
        self.time_dim = 1 if self.rnn.batch_first else 0
        self.dtype = next(self.rnn.parameters()).dtype
        # reduce obj defaults to PCA, this is different than axes
        self.reduce_obj = PCA(n_components=2)
        self.axes = None
        self.inverse_axes = None

        if fit_states is not None:
            # if not done here, do on the trajectory in find_flow
            self._fit_traj(self.fit_states)

        if axes is not None:
            # assert two axes are present
            assert axes.dim() == 2 and axes.shape[0] == 2
            self.axes = axes.to(dtype=self.dtype)
            self.inverse_axes = torch.linalg.pinv(self.axes)

    def transform(self, x):
        """Project packed samples [N, D] into coordinates [N, 2].

        Uses explicit axes when supplied; otherwise fits PCA on x if not yet
        fitted, then transforms x. Returns a tensor for explicit axes and a
        NumPy array for PCA. LSTM tuples must be packed before calling.
        """
        if self.axes is not None:
            return x @ self.axes.T
        else:
            try:
                check_is_fitted(self.reduce_obj)
            except NotFittedError:
                self._fit_traj(x)
            return self.reduce_obj.transform(x)

    def inverse_transform(self, x):
        """Reconstruct packed states [N, D] from coordinates [N, 2].

        Uses the explicit-axis pseudoinverse or fitted PCA inverse. Returns
        a tensor for axes and a NumPy array for PCA. For LSTMs, reconstructed
        states contain both h and c; this does not return a tuple.
        """
        if self.inverse_axes is not None:
            return x @ self.inverse_axes.T
        else:
            return self.reduce_obj.inverse_transform(x)

    def find_nonlinear_flow(self, *args, **kwargs) -> list:
        """Compute 2D flow fields in a region subspace along a trajectory."""
        raise NotImplementedError

    def find_linear_flow(self, *args, **kwargs) -> list:
        """Compute linearized flow fields in a 2D subspace."""
        raise NotImplementedError

    @staticmethod
    def _nxd(x: torch.Tensor) -> torch.Tensor:
        """Flatten tensor [..., D] to [N, D], treating [D] as one sample.

        This helper does not pack LSTM tuples. All leading dimensions are
        sample dimensions; native layer dimensions must be handled by callers.
        """
        if x.dim() == 1:
            x = x.unsqueeze(0)
        x = torch.flatten(x, end_dim=-2)
        return x

    def _fit_traj(self, trajectory: torch.Tensor):
        """Fit two-component PCA on detached CPU packed states [..., D].

        Flattens leading dimensions. LSTM tuples must be packed first, so
        PCA sees all 2H state features. Requires at least two samples/features.
        """
        # Gather activity for specified region and cell type
        temp_act = torch.reshape(trajectory, (-1, trajectory.shape[-1]))
        # Do PCA on the specified region(s)
        self.reduce_obj.fit(temp_act)

    def _reduce_traj(self, trajectory: torch.Tensor) -> torch.Tensor:
        """Flatten and project packed states [..., D] to a tensor [N, 2].

        Uses explicit axes or PCA (fitting it on these states if necessary).
        The result uses the model dtype. LSTM tuples are not accepted here.
        """
        # Gather activity for specified region and cell type
        temp_act = torch.reshape(trajectory, (-1, trajectory.shape[-1]))
        reduced_traj = self.transform(temp_act)
        reduced_traj = torch.as_tensor(reduced_traj, dtype=self.dtype)

        return reduced_traj

    def _inverse_grid(
        self,
        lower_bound_x: float,
        upper_bound_x: float,
        lower_bound_y: float,
        upper_bound_y: float,
        expand_dims: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Create a two-dimensional grid and reconstruct full packed states.

        Args:
            lower_bound_x: Minimum first-axis coordinate.
            upper_bound_x: Maximum first-axis coordinate.
            lower_bound_y: Minimum second-axis coordinate.
            upper_bound_y: Maximum second-axis coordinate.
            expand_dims: Return grid-shaped arrays instead of flattened samples.

        Returns:
            Coordinates [P*P, 2] and packed states [P*P, D], where P is
            num_points. With expand_dims, shapes are [P, P, 2] and [P, P, D].
            LSTM reconstructed states use [h, c] order.
        """
        # Num points is along each axis, not in total
        x = torch.linspace(lower_bound_x, upper_bound_x, self.num_points)
        y = torch.linspace(lower_bound_y, upper_bound_y, self.num_points)

        # Gather 2D grid for flow fields
        xv, yv = torch.meshgrid(x, y, indexing="ij")
        xv, yv = xv.unsqueeze(-1), yv.unsqueeze(-1)

        # Convert the grid to a tensor and flatten for PCA
        low_dim_grid = torch.cat((xv, yv), dim=-1)
        low_dim_grid = torch.flatten(low_dim_grid, start_dim=0, end_dim=1)

        # Inverse PCA to input grid into network
        inverse_grid = self.inverse_transform(low_dim_grid)
        inverse_grid = torch.as_tensor(inverse_grid, dtype=self.dtype)

        if expand_dims:
            low_dim_grid = torch.reshape(
                low_dim_grid, (self.num_points, self.num_points, 2)
            )
            inverse_grid = torch.reshape(
                inverse_grid, (self.num_points, self.num_points, inverse_grid.shape[-1])
            )

        return low_dim_grid, inverse_grid

    def _compute_velocity(
        self, h_next: torch.Tensor, h_prev: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return x/y one-step displacements between projected [..., 2] states."""
        x_vel = h_next[..., 0] - h_prev[..., 0]
        y_vel = h_next[..., 1] - h_prev[..., 1]
        return x_vel, y_vel

    def _compute_speed(self, x_vel: torch.Tensor, y_vel: torch.Tensor) -> torch.Tensor:
        """Return Euclidean projected speed divided by its maximum in this field.

        This is a relative visualization scale, not the full-state residual.
        The current normalization is undefined for an entirely zero-speed field.
        """
        speed = torch.sqrt(x_vel**2 + y_vel**2)
        return speed / speed.max()

    def _reshape_vals(
        self,
        x_vels: torch.Tensor,
        y_vels: torch.Tensor,
        grid: torch.Tensor,
        speeds: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Broadcast data to FlowField format

        Args:
            x_vels (Tensor): x velocities
            y_vels (Tensor): y velocities
            grid (Tensor): low dimensional grid coordinates
            speeds (Tensor): magnitude of x_vels and y_vels
        """
        # Reshape to match FlowField object requirements
        x_vels = torch.reshape(x_vels, (self.num_points, self.num_points))
        y_vels = torch.reshape(y_vels, (self.num_points, self.num_points))
        grid = torch.reshape(grid, (self.num_points, self.num_points, 2))
        speeds = torch.reshape(speeds, (self.num_points, self.num_points))
        return x_vels, y_vels, grid, speeds

    def _set_bounds(self) -> Tuple[float, float, float, float]:
        """
        Design grid bounds using a static center and offsetting
        """
        lower_bound_x = self.x_center - self.x_offset
        upper_bound_x = self.x_center + self.x_offset
        lower_bound_y = self.y_center - self.y_offset
        upper_bound_y = self.y_center + self.y_offset
        return lower_bound_x, upper_bound_x, lower_bound_y, upper_bound_y

    def _set_tv_bounds(self, traj: torch.Tensor) -> Tuple[float, float, float, float]:
        """Return grid bounds centered on a projected two-component state.

        Args:
            traj: Reduced state [2], not a full hidden or packed LSTM state.

        Returns:
            Lower/upper x bounds and lower/upper y bounds, rounded to one decimal.
        """
        lower_bound_x = torch.round(traj[0] - self.x_offset, decimals=1).item()
        upper_bound_x = torch.round(traj[0] + self.x_offset, decimals=1).item()
        lower_bound_y = torch.round(traj[1] - self.y_offset, decimals=1).item()
        upper_bound_y = torch.round(traj[1] + self.y_offset, decimals=1).item()
        return lower_bound_x, upper_bound_x, lower_bound_y, upper_bound_y
