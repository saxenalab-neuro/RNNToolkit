import torch
import torch.nn as nn
import time

from .fp import FixedPointCollection
from typing import Generic, TypeVar

RNN = TypeVar("RNN", bound=nn.Module)


class FixedPointFinderBase(Generic[RNN]):
    def __init__(
        self,
        rnn: RNN,
        verbose: bool = False,
        **kwargs,
    ):
        """Initialize tensor-based helpers for a concrete fixed-point finder.

        Args:
            rnn: Recurrent module with parameters and a batch_first attribute.
            verbose: Whether to print progress messages.
            **kwargs: Reserved for subclasses.

        Note:
            Sampling, broadcasting, and distance helpers require packed tensors.
            Native LSTM tuples are handled by the concrete find_fixed_points method.
        """

        self.device = next(rnn.parameters()).device
        self.verbose = verbose

        self.rnn = rnn
        self.batch_first = self.rnn.batch_first

    # *************************************************************************
    # Primary exposed functions ***********************************************
    # *************************************************************************

    def sample_states(
        self,
        state_traj: torch.Tensor,
        n_inits: int,
        noise_scale: float = 0.0,
        exclude_zero_tensors: bool = False,
    ) -> torch.Tensor:
        """Sample full states from recorded trajectories, with optional noise.

        Args:
            state_traj: Tensor [D] or [..., D]. Pack LSTM (h, c) trajectories
                first so corresponding hidden and cell states are sampled together.
            n_inits: Number of initial states sampled with replacement.
            noise_scale: Standard deviation of independent Gaussian noise;
                use a non-negative value. Zero leaves samples unperturbed.
            exclude_zero_tensors: Exclude rows whose every component is zero.

        Returns:
            Packed initial guesses [n_inits, D]. Leading trajectory dimensions
            are flattened before sampling. There must be at least one eligible row.
        """
        if state_traj.dim() == 1:
            state_traj = state_traj.unsqueeze(0)

        # Get the batch shape of state trajectory, assumes -1 is state dim
        flat_state_traj = torch.flatten(state_traj, end_dim=-2)

        if exclude_zero_tensors:
            non_zero_rows, _ = torch.nonzero(flat_state_traj, as_tuple=True)
            non_zero_rows = torch.unique(non_zero_rows)
            flat_state_traj = flat_state_traj[non_zero_rows]

        rand_indices = torch.randint(high=flat_state_traj.shape[0], size=(n_inits,))
        states = flat_state_traj[rand_indices]

        # Add IID Gaussian noise to the sampled states
        states = self._add_gaussian_noise(states, noise_scale)

        assert not torch.any(torch.isnan(states)), (
            "Detected NaNs in sampled states. Check state_traj and valid_bxt."
        )

        return states

    def find_fixed_points(self, *args, **kwargs):
        """Overwritten by subclass for network specific optimizations"""
        raise NotImplementedError

    # *************************************************************************
    # Helper functions ********************************************************
    # *************************************************************************

    def _add_gaussian_noise(
        self, data: torch.Tensor, noise_scale: float = 0.0
    ) -> torch.Tensor:
        """Add independent Gaussian noise to a tensor without unpacking its states.

        Args:
            data: Tensor of packed samples.
            noise_scale: Non-negative noise standard deviation; zero returns data.

        Returns:
            Tensor of the same shape. Noise currently uses CPU/default-dtype
            allocation; the documented example workflow uses float32 CPU states.
        """

        # Add IID Gaussian noise
        if noise_scale == 0.0:
            return data  # no noise to add
        else:
            return data + noise_scale * torch.randn(*data.shape)

    @staticmethod
    def identify_q_outliers(fps: FixedPointCollection, q_thresh: float) -> torch.Tensor:
        """Identify fixed points with optimized q values that exceed a
        specified threshold.

        Args:
            fps: A FixedPoints object containing optimized fixed points and
                associated metadata.
            q_thresh: A scalar float indicating the threshold on fixed
                points' q values.

        Returns:
            A tensor containing the indices into fps corresponding to
            the fixed points with q values exceeding the threshold.

        Usage:
            idx = identify_q_outliers(fps, q_thresh)
            outlier_fps = fps[idx]
        """
        assert fps.qstar is not None
        return torch.where(fps.qstar > q_thresh)[0]

    @staticmethod
    def identify_q_non_outliers(
        fps: FixedPointCollection, q_thresh: float
    ) -> torch.Tensor:
        """Identify fixed points with optimized q values that do not exceed a
        specified threshold.

        Args:
            fps: A FixedPoints object containing optimized fixed points and
                associated metadata.
            q_thresh: A scalar float indicating the threshold on fixed points'
                q values.

        Returns:
            A tensor containing the indices into fps corresponding to the
            fixed points with q values that do not exceed the threshold.

        Usage:
            idx = identify_q_non_outliers(fps, q_thresh)
            non_outlier_fps = fps[idx]
        """
        assert fps.qstar is not None
        return torch.where(fps.qstar <= q_thresh)[0]

    @staticmethod
    def get_init_non_distance_outliers(
        initial_states: torch.Tensor, dist_thresh: float
    ) -> torch.Tensor:
        """
        get initial states that are far from centroid based on threshold.

        Args:
            initial_states (Tensor): initial states of fp optimization [n, state_dim]
            dist_thresh (float): Threshold from initial states which is far.

        Returns:
            init_non_outlier_idx (Tensor): indices to initial_states tensor inside threshold
        """

        # Centroid of initial_states, shape (n_states,)
        centroid = torch.mean(initial_states, dim=0)

        # Distance of each initial state from the centroid, shape (n,)
        init_dists = torch.linalg.norm(initial_states - centroid, axis=1)
        avg_init_dist = torch.mean(init_dists)

        # Normalized distances of initial states to the centroid, shape: (n,)
        scaled_init_dists = torch.true_divide(init_dists, avg_init_dist)

        init_non_outlier_idx = torch.where(scaled_init_dists < dist_thresh)[0]
        return init_non_outlier_idx

    @staticmethod
    def get_fp_non_distance_outliers(
        fps: FixedPointCollection, initial_states: torch.Tensor, dist_thresh: float
    ) -> torch.Tensor:
        """
        get fixed points that are far from initial states based on threshold.

        Args:
            fps (FixedPointCollection): fps discovered [n, state_dim]
            initial_states (Tensor): initial states of optimization [n, state_dim]
            dist_thresh (float): threshold at which fixed points are considered far

        Returns:
            fsp_non_outlier_distance (Tensor): indices to fps object that are not far
        """
        # TODO make sure there is a warning if all fps are removed during this function
        # Centroid of initial_states, shape (n_states,)
        centroid = torch.mean(initial_states, dim=0)

        # Distance of each initial state from the centroid, shape (n,)
        init_dists = torch.linalg.norm(initial_states - centroid, axis=1)
        avg_init_dist = torch.mean(init_dists)

        # Distance of each FP from the initial_states centroid
        fps_dists = torch.linalg.norm(fps.xstar - centroid, axis=1)

        # Normalized
        scaled_fps_dists = torch.true_divide(fps_dists, avg_init_dist)

        fps_non_outlier_idx = torch.where(scaled_fps_dists < dist_thresh)[0]
        return fps_non_outlier_idx

    def _broadcast_nxd(self, data: torch.Tensor, tile_n: int = 1) -> torch.Tensor:
        """
        Takes in a tensor of shape [..., d] and reshapes to nxd
        tiles by tile_n if 1D
        """
        # Broadcast to [n, d]
        if data.dim() == 1:
            # If only 1d, then tile
            data = torch.tile(data, [tile_n, 1])
        else:
            # If > 1d, then flatten up to last dim
            data = torch.flatten(data, end_dim=-2)
        # Ensure proper device and dtype
        data = data.to(self.device)
        return data

    def _print_if_verbose(self, *args, **kwargs):
        if self.verbose:
            print(*args, **kwargs)

    @classmethod
    def _print_iter_update(
        cls,
        iter_count: int,
        t_start: float,
        q: torch.Tensor,
        dq: torch.Tensor,
        lr: float,
        is_final: bool = False,
    ):
        t = time.time()
        t_elapsed = t - t_start
        avg_iter_time = t_elapsed / iter_count

        if is_final:
            delimiter = "\n\t\t"
            print("\t\t%d iters%s" % (iter_count, delimiter), end="")
        else:
            delimiter = ", "
            print("\tIter: %d%s" % (iter_count, delimiter), end="")

        if q.size == 1:
            print("q = %.2e%sdq = %.2e%s" % (q, delimiter, dq, delimiter), end="")
        else:
            mean_q = torch.mean(q)
            std_q = torch.std(q)

            mean_dq = torch.mean(dq)
            std_dq = torch.std(dq)

            print(
                "q = %.2e +/- %.2e%s"
                "dq = %.2e +/- %.2e%s"
                % (mean_q, std_q, delimiter, mean_dq, std_dq, delimiter),
                end="",
            )

        print("avg iter time = %.2e sec" % avg_iter_time, end="")
