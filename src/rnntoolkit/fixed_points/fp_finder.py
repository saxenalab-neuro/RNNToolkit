import torch
import torch.nn as nn
import numpy as np
import time
from copy import deepcopy

from .fp import FixedPointCollection
from rnntoolkit.fixed_points.fp_finder_base import FixedPointFinderBase


class FixedPointFinder(FixedPointFinderBase):
    def __init__(
        self,
        rnn: nn.RNN,
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
        super().__init__(rnn, verbose)
        """Creates a FixedPointFinder object.
        Inherited from FixedPointFinderBase

        Optimization terminates once every initialization satisfies one or
        both of the following criteria:
            1. q < tol_q
            2. dq < tol_dq * learning_rate

        Args:
            rnn_cell: A Pytorch RNN
            tol_q (optional): A positive scalar specifying the optimization
                termination criteria on each q-value. Default: 1e-12.
            tol_dq (optional): A positive scalar specifying the optimization
                termination criteria on the improvement of each q-value (i.e.,
                "dq") from one optimization iteration to the next.
            max_iters (optional): A non-negative integer specifying the
                maximum number of gradient descent iterations allowed.
            do_rerun_q_outliers (optional): A bool indicating whether or not
                to run additional optimization iterations on putative outlier
                states
            outlier_q_scale (optional): A positive float specifying the q
                value for putative outlier fixed points, relative to the median q
                value across all identified fixed points. Default: 10.
            do_exclude_distance_outliers (optional): A bool indicating
                whether or not to discard states that are far away from the set
                of initial states
            outlier_distance_scale (optional): A positive float specifying a
                normalized distance cutoff used to exclude distance outliers
            tol_unique (optional): A positive scalar specifying the numerical
                precision required to label two fixed points as being unique from
                one another. 
            max_n_unique (optional): A positive integer indicating the max
                number of unique fixed points to keep.
            dtype: string indicating the data type to use for all numerical ops
                and objects. Default: 'float32'
            random_seed: Seed for numpy random number generator. Default: 0.
            verbose (optional): A bool indicating whether to print high-level
                status updates. Default: True.
            super_verbose (optional): A bool indicating whether or not to
                print per-iteration updates during each optimization. Default:
                False.
            n_iters_per_print_update (optional): An int specifying how often
                to print updates during the fixed point optimizations. Default:
                100.
        """

        self.dtype = dtype
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
        initial_states: torch.Tensor,
        ext_inputs: torch.Tensor,
        n_rounds_q_opt: int = 1,
    ) -> tuple[FixedPointCollection, FixedPointCollection]:
        """Finds RNN fixed points and the Jacobians at the fixed points.

        Args:
            initial_states: Tensor specifying the initial
                states of the RNN, from which the optimization will search for
                fixed points.
            ext_inputs: external inputs to the RNN
            n_rounds_q_opt: Number of rounds to run extra iterations on q
            outliers

        Returns:
            unique_fps: A FixedPoints object containing the set of unique
                fixed points after optimizing from all initial_states
            all_fps: A FixedPoints object containing the likely redundant set
                of fixed points (and associated metadata) resulting from ALL
                initializations in initial_states
        """

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
        """Removes putative distance outliers from a set of fixed points.
        See docstring for identify_distance_non_outliers(...).
        """

        idx_keep = self.get_fp_non_distance_outliers(
            fps, initial_states, self.outlier_distance_scale
        )
        return fps[idx_keep.tolist()]

    def _run_additional_iterations_on_outliers(
        self, fps: FixedPointCollection, n_rounds: int = 1
    ) -> FixedPointCollection:
        """Detects outlier states with respect to the q function and runs
        additional optimization iterations on those states This should only be
        used after calling either _run_joint_optimization or
        _run_sequential_optimizations.

        Args:
            fps: A FixedPoints object containing (partially) optimized
                fixed points and associated metadata.
            stim_inp: additional stimulus to give network during optimization
            W_rec: replaces self.mrnn.W_rec during forward pass
            W_inp: replaces self.mrnn.W_inp during forward pass

        Returns:
            A FixedPoints object containing the further-optimized fixed points
            and associated metadata.
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
        """Finds multiple fixed points via a joint optimization over multiple
        state vectors.

        Args:
            initial_states: Tensor specifying the initial
                states of the RNN, from which the optimization will search for
                fixed points.
            ext_inp: Tensor specifying a set of constant
                inputs into the RNN.

        Returns:
            fps: A FixedPoints object containing the optimized fixed points
            and associated metadata.
        """

        # Get batch and time dims
        if self.batch_first:
            TIME_DIM = 1
        else:
            TIME_DIM = 0

        initial_states = self._broadcast_nxd(initial_states, tile_n=1)

        # Get batch size of states
        n = initial_states.shape[0]

        # Broadcast external input to [n, 1, d]
        ext_inp = self._broadcast_nxd(ext_inp, tile_n=n)
        ext_inp = ext_inp.unsqueeze(TIME_DIM)

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

            # currently only works for 1 layer rnns
            _, F_x_1xbxd = self.rnn(
                ext_inp,
                h.unsqueeze(0),
            )
            F_x_1xbxd = F_x_1xbxd.squeeze(0)

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

        xstar = initial_states.detach().cpu()
        F_xstar = F_x_1xbxd.detach().cpu()

        # Indicate same n_iters for each initialization (i.e., joint optimization)
        n_iters = torch.tile(torch.tensor([iter_count]), dims=(F_xstar.shape[0],))
        inputs_bxd = ext_inp.squeeze(TIME_DIM)

        fps = FixedPointCollection(
            xstar=xstar,
            x_init=initial_states,
            inputs=inputs_bxd,
            F_xstar=F_xstar,
            qstar=ev_q_b,
            dq=ev_dq_b,
            n_iters=n_iters,
            tol_unique=self.tol_unique,
            dtype=self.torch_dtype,
        )

        return fps
