import torch
import torch.nn as nn
import numpy as np
import time
from copy import deepcopy

from .fp import FixedPointCollection
from rnntoolkit.adapter import RNNAdapter
from rnntoolkit.fixed_points.fp_finder_base import FixedPointFinderBase


class FixedPointFinder(FixedPointFinderBase):
    def __init__(
        self,
        rnn: nn.RNN | nn.GRU | nn.LSTM,
        lr_init: float = 1e-4,
        tol_q: float = 1e-12,
        tol_dq: float = 1e-20,
        max_iters: int = 5000,
        do_rerun_q_outliers: bool = False,
        outlier_q_scale: float = 10.0,
        do_exclude_distance_outliers: bool = True,
        outlier_distance_scale: float = 10.0,
        tol_unique: float = 1e-3,
        max_n_unique: float = np.inf,
        dtype: str = "float32",
        random_seed: int = 0,
        verbose: bool = False,
        super_verbose: bool = False,
        n_iters_per_print_update: int = 100,
    ):
        """Configure fixed-point optimization for RNN, GRU, or LSTM dynamics.

        Optimization stops when each guess meets q < tol_q or
        dq < tol_dq * learning_rate, or the iteration limit is reached.

        Args:
            rnn: Single-layer, unidirectional nn.RNN, nn.GRU, or nn.LSTM
                without projections. Either batch_first setting is supported.
            lr_init: Initial Adam learning rate; default 1e-4.
            tol_q: Residual objective convergence threshold; default 1e-12.
            tol_dq: Objective-change threshold scaled by learning rate;
                default 1e-20.
            max_iters: Maximum optimization iterations; use a positive integer.
            do_rerun_q_outliers: Whether to rerun optimization for high-q candidates.
            outlier_q_scale: Multiplier on median q defining outliers for reruns.
            do_exclude_distance_outliers: Whether to filter candidates far from
                the original initial-state centroid.
            outlier_distance_scale: Distance threshold normalized by the mean
                initial-state distance from that centroid.
            tol_unique: State/input distance tolerance for identifying duplicates.
            max_n_unique: Maximum unique points retained; infinity keeps all.
            dtype: Torch dtype name recorded as collection metadata, default
                'float32'. Supply states/inputs matching model dtype and device.
            random_seed: Seed for NumPy subsampling of excess unique points.
                State sampling uses PyTorch's global random generator.
            verbose: Print high-level progress when True; default False.
            super_verbose: Print iteration updates when True and verbose is enabled.
            n_iters_per_print_update: Number of iterations between progress updates.
        """
        super().__init__(rnn, verbose)

        self.dtype = dtype
        self.adapter = RNNAdapter(rnn)
        self.device = next(rnn.parameters()).device
        self.torch_dtype = getattr(torch, self.dtype)

        # Make random sequences reproducible
        self.random_seed = random_seed
        self.rng = np.random.RandomState(random_seed)

        # *********************************************************************
        # Optimization hyperparameters ****************************************
        # *********************************************************************

        self.lr_init = lr_init
        self.tol_q = tol_q
        self.tol_dq = tol_dq
        self.max_iters = max_iters
        self.do_rerun_q_outliers = do_rerun_q_outliers
        self.outlier_q_scale = outlier_q_scale
        self.do_exclude_distance_outliers = do_exclude_distance_outliers
        self.outlier_distance_scale = outlier_distance_scale
        self.tol_unique = tol_unique
        self.max_n_unique = max_n_unique
        self.super_verbose = super_verbose
        self.n_iters_per_print_update = n_iters_per_print_update

    # *************************************************************************
    # Primary exposed functions ***********************************************
    # *************************************************************************

    def find_fixed_points(
        self,
        initial_states: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        ext_inputs: torch.Tensor,
        n_rounds_q_opt: int = 1,
    ) -> tuple[FixedPointCollection, FixedPointCollection]:
        """Optimize full states for nearly stationary one-step dynamics.

        Minimizes q(s) = 0.5 * ||F(u, s) - s|| squared at a fixed input.
        For LSTMs this objective includes both hidden and cell residuals.

        Args:
            initial_states: Initial guesses [N, D] (or a single [D] state),
                with D = H for RNN/GRU and D = 2H for LSTM. LSTMs also
                accept (h, c) tuples with matching [N, H] components.
            ext_inputs: Shared constant input [I], or one input per guess
                [N, I]. Each guess's input is held fixed during optimization.
            n_rounds_q_opt: Maximum extra rounds for q outliers when enabled
                by do_rerun_q_outliers.

        Returns:
            Pair (unique_fps, all_fps) of FixedPointCollection objects.
            xstar, x_init, and F_xstar are packed [N, D] tensors, including
            [h, c] for LSTMs. qstar is the residual objective evaluated at
            the returned xstar. Filtering may leave no unique points.
            Jacobians and stability are not computed here; use Linearization.

        Note:
            Reaching max_iters or a small objective change does not guarantee
            a small residual; inspect qstar. The finder currently disables
            requires_grad on model parameters and does not restore those flags.
            Caller-provided initial state values are not modified.
        """

        if not self.adapter.is_packed(initial_states):
            initial_states = self.adapter.pack_state(initial_states)

        all_fps = self._fp_optimization(
            initial_states,
            ext_inputs,
        )

        # Filter out duplicates after from the first optimization round
        unique_fps = all_fps.get_unique()

        self._print_if_verbose("\tIdentified %d unique fixed points." % unique_fps.n)

        if self.do_exclude_distance_outliers:
            unique_fps = self._exclude_distance_outliers(unique_fps, initial_states)

        # Optionally run additional optimization iterations on identified
        # fixed points with q values on the large side of the q-distribution.
        if self.do_rerun_q_outliers:
            unique_fps = self._run_additional_iterations_on_outliers(
                unique_fps,
                n_rounds=n_rounds_q_opt,
            )
            # Filter out duplicates after from the second optimization round
            unique_fps = unique_fps.get_unique()

        # Optionally subselect from the unique fixed points (e.g., for
        # computational savings when not all are needed.)
        if unique_fps.n > self.max_n_unique:
            self._print_if_verbose(
                "\tRandomly selecting %d unique "
                "fixed points to keep." % self.max_n_unique
            )
            max_n_unique = int(self.max_n_unique)
            idx_keep = list(self.rng.choice(unique_fps.n, max_n_unique, replace=False))
            unique_fps = unique_fps[idx_keep]

        self._print_if_verbose("\tFixed point finding complete.\n")

        return unique_fps, all_fps

    # *************************************************************************
    # Helper functions ********************************************************
    # *************************************************************************

    def _exclude_distance_outliers(
        self, fps: FixedPointCollection, initial_states: torch.Tensor
    ) -> FixedPointCollection:
        """Filter packed fixed points by distance from the initial-state centroid.

        Args:
            fps: Candidate FixedPointCollection with packed states.
            initial_states: Packed initial guesses [N, D].

        Returns:
            Collection retaining states within the configured normalized
            distance threshold. Distances include both h and c for LSTMs.
        """

        idx_keep = self.get_fp_non_distance_outliers(
            fps, initial_states, self.outlier_distance_scale
        )
        return fps[idx_keep.tolist()]

    def _run_additional_iterations_on_outliers(
        self, fps: FixedPointCollection, n_rounds: int = 1
    ) -> FixedPointCollection:
        """Reoptimize candidates whose q exceeds the configured median-based cutoff.

        Args:
            fps: Collection containing packed states, inputs, qstar, and n_iters.
            n_rounds: Maximum number of additional optimization rounds.

        Returns:
            Collection with outlier states and optimization metadata updated.
            LSTM states remain packed throughout reruns.
        """

        assert fps.qstar is not None

        outlier_min_q = float(np.median(fps.qstar) * self.outlier_q_scale)

        def perform_outlier_optimization(
            fps: FixedPointCollection,
        ) -> FixedPointCollection:
            idx_outliers = self.identify_q_outliers(fps, outlier_min_q)

            outlier_fps = fps[idx_outliers.tolist()]
            n_prev_iters = outlier_fps.n_iters
            inputs = outlier_fps.inputs
            initial_states = outlier_fps.xstar

            self._print_if_verbose(
                "\tPerforming another round of "
                "joint optimization, "
                "over outlier states only."
            )

            assert inputs is not None
            assert n_prev_iters is not None

            updated_outlier_fps = self._fp_optimization(initial_states, inputs)

            assert updated_outlier_fps.n_iters is not None

            updated_outlier_fps.n_iters += n_prev_iters
            fps[idx_outliers.tolist()] = updated_outlier_fps

            return fps

        def outlier_update(fps: FixedPointCollection) -> torch.Tensor:
            idx_outliers = self.identify_q_outliers(fps, outlier_min_q)
            n_outliers = len(idx_outliers)

            self._print_if_verbose(
                "\n\tDetected %d putative outliers "
                "(q>%.2e)." % (n_outliers, outlier_min_q)
            )

            return idx_outliers

        idx_outliers = outlier_update(fps)

        if len(idx_outliers) == 0:
            return fps

        for _ in range(n_rounds):
            fps = perform_outlier_optimization(fps)
            idx_outliers = outlier_update(fps)
            if len(idx_outliers) == 0:
                return fps

        return fps

    def _fp_optimization(
        self,
        initial_states: torch.Tensor,
        ext_inp: torch.Tensor,
    ) -> FixedPointCollection:
        """Optimize a batch of packed state guesses using Adam.

        Args:
            initial_states: Packed initial states [N, D], including [h, c]
                for LSTM. Cloned into a separate optimization tensor.
            ext_inp: Constant input [I] or inputs [N, I].

        Returns:
            FixedPointCollection containing every optimized state. xstar,
            F_xstar, qstar, and dq describe the final optimizer step; x_init
            preserves the original guesses. The adapter handles native
            hidden/cell shapes and the model's batch_first setting.
        """

        initial_states = self._broadcast_nxd(initial_states, tile_n=1).detach().clone()
        x_init = initial_states.clone()

        # Get batch size of states
        n = initial_states.shape[0]

        # The adapter expects [batch, features] for both input and state.
        ext_inp = self._broadcast_nxd(ext_inp, tile_n=n).detach()

        # assert the correct batch shapes
        assert ext_inp.shape[0] == initial_states.shape[0]

        self._print_if_verbose(
            "\nSearching for fixed points from %d initial states.\n" % n
        )

        # Ensure that fixed point optimization does not alter RNN parameters.
        print(
            "\tFreezing model parameters so model is not affected by fixed point optimization."
        )

        for p in self.rnn.parameters():
            p.requires_grad = False

        ext_inp.requires_grad = False
        initial_states.requires_grad = True

        self._print_if_verbose("\tFinding fixed points via joint optimization.")
        init_lr = self.lr_init
        optimizer = torch.optim.Adam([initial_states], lr=self.lr_init)

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.25, patience=10, cooldown=0, threshold=1e-10
        )

        iter_count = 1
        iter_learning_rate = init_lr
        t_start = time.time()
        q_prev_b = torch.full((n,), float("nan"), device=self.device)

        while True:
            h = initial_states.clone()

            F_x_1xbxd = self.adapter(ext_inp, h)

            dx_bxd = h - F_x_1xbxd
            q_b = 0.5 * torch.sum(torch.square(dx_bxd), dim=-1)
            q_scalar = torch.mean(q_b)
            dq_b = torch.abs(q_b - q_prev_b)

            optimizer.zero_grad()
            q_scalar.backward()

            optimizer.step()
            scheduler.step(metrics=q_scalar.detach())

            iter_learning_rate = scheduler.state_dict()["_last_lr"][0]

            ev_q_b = q_b.detach().cpu()
            ev_dq_b = dq_b.detach().cpu()

            if (
                self.super_verbose
                and np.mod(iter_count, self.n_iters_per_print_update) == 0
            ):
                self._print_iter_update(
                    iter_count, t_start, ev_q_b, ev_dq_b, iter_learning_rate
                )

            if iter_count > 1 and torch.all(
                torch.logical_or(
                    ev_dq_b < self.tol_dq * iter_learning_rate, ev_q_b < self.tol_q
                )
            ):
                """Here dq is scaled by the learning rate. Otherwise very
                small steps due to very small learning rates would spuriously
                indicate convergence. This scaling is roughly equivalent to
                measuring the gradient norm."""
                self._print_if_verbose("\tOptimization complete to desired tolerance.")
                break

            if iter_count + 1 > self.max_iters:
                self._print_if_verbose(
                    "\tMaximum iteration count reached. Terminating."
                )
                break

            q_prev_b = q_b
            iter_count += 1

        if self.verbose:
            self._print_iter_update(
                iter_count, t_start, ev_q_b, ev_dq_b, iter_learning_rate, is_final=True
            )

        # Report the map and residual at the returned (post-optimizer) state.
        with torch.no_grad():
            final_next = self.adapter(ext_inp, initial_states)
            final_q = 0.5 * (initial_states - final_next).square().sum(dim=-1)
            ev_dq_b = (final_q - q_b.detach()).abs().cpu()
            ev_q_b = final_q.cpu()
        xstar = initial_states.detach().cpu()
        F_xstar = final_next.cpu()

        # Indicate same n_iters for each initialization (i.e., joint optimization)
        n_iters = torch.tile(torch.tensor([iter_count]), dims=(F_xstar.shape[0],))
        inputs_bxd = ext_inp

        fps = FixedPointCollection(
            xstar=xstar,
            x_init=x_init.cpu(),
            inputs=inputs_bxd,
            F_xstar=F_xstar,
            qstar=ev_q_b,
            dq=ev_dq_b,
            n_iters=n_iters,
            tol_unique=self.tol_unique,
            dtype=self.torch_dtype,
        )

        return fps
