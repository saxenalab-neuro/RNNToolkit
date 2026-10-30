import torch
import torch.nn as nn
from rnntoolkit.adapter import RNNAdapter
from typing import Tuple


class Linearization:
    """Linearize one-step RNN, GRU, or LSTM dynamics.

    State width D is H for RNN/GRU and 2H for LSTM. LSTM states may be
    packed [h, c] tensors or (h, c) tuples; all state outputs are packed.
    Jacobians include the full hidden/cell coupling for LSTMs.
    """

    def __init__(
        self,
        rnn: nn.RNN | nn.GRU | nn.LSTM,
        input: torch.Tensor,
        state: torch.Tensor,
        gramian_k: int = 10,
        gramian_tol: float = 1e-3,
    ):
        """Initialize the differentiable one-step adapter.

        Args:
            rnn: Single-layer, unidirectional nn.RNN, nn.GRU, or nn.LSTM.
                Projected LSTMs are not supported. Tensors must match the
                module's device and dtype.
        """
        assert state.dim() == 1
        assert input.dim() == 1

        self.rnn = rnn
        self.adapter = RNNAdapter(self.rnn)
        self.input = input
        self.state = state
        self.jac_rec, self.jac_inp = self._jacobian()
        self.jac_rec_real_eigs, self.jac_rec_im_eigs, self.jac_rec_eigvecs = (
            self._eigendecomposition()
        )
        self.gramian_k = gramian_k
        self.gramian_tol = gramian_tol
        self.ctrl_gramian = self._controllability_gramian()

    def __call__(
        self,
        delta_input: torch.Tensor,
        delta_h: torch.Tensor,
    ) -> torch.Tensor:
        """Evaluate the affine approximation; alias for :meth:`forward`.

        Args:
            input: Reference input [I].
            state: Reference state [D], or LSTM pair of [H] tensors.
            delta_input: Input perturbation [I].
            delta_h: State perturbations [D] or [..., D], or an LSTM tuple
                with matching [..., H] components; passed as delta_state.

        Returns:
            Packed predicted next states [D] or [N, D], with leading
            perturbation dimensions flattened into N samples.
        """
        return self.forward(delta_input, delta_h)

    def forward(
        self,
        delta_input: torch.Tensor,
        delta_state: torch.Tensor | Tuple,
    ) -> torch.Tensor:
        """Evaluate F(u, s) + J_state delta_state + J_input delta_input.

        Args:
            input: Reference input [I].
            state: Reference state [D], or LSTM tuple (h, c) of [H] tensors.
            delta_input: Shared input perturbation [I].
            delta_state: State perturbations [D] or [..., D]. LSTMs also
                accept (delta_h, delta_c) with matching [..., H] components.

        Returns:
            Packed predicted next state [D], or [N, D] for batched
            perturbations. Both h and c are advanced for LSTMs.
        """

        # pack states if not done already
        if not self.adapter.is_packed(self.state):
            self.state = self.adapter.pack_state(self.state)

        if not self.adapter.is_packed(delta_state):
            delta_state = self.adapter.pack_state(delta_state)

        # Assert correct shapes
        assert delta_input.dim() == 1

        # assert correct batch dimensions
        if delta_state.dim() > 1:
            delta_state = delta_state.flatten(start_dim=0, end_dim=-2)

        with torch.no_grad():
            # Get h_next for affine function
            next_state = self.adapter(self.input, self.state)

        h_pert = (
            next_state + delta_state @ self.jac_rec.T + (self.jac_inp @ delta_input)
        )

        return h_pert

    def _jacobian(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Differentiate the full one-step state update at (input, state).

        Args:
            input: Reference input [I].
            state: Reference state [D], or LSTM tuple (h, c) of [H] tensors.

        Returns:
            Pair (J_state, J_input) with shapes [D, D] and [D, I]. Matrix
            dimensions are retained when D or I is one. For LSTMs D = 2H;
            the blocks describe derivatives of both next h and next c
            with respect to both current h and current c.
        """

        if not self.adapter.is_packed(self.state):
            self.state = self.adapter.pack_state(self.state)

        assert self.state.dim() == 1
        assert self.input.dim() == 1

        with torch.no_grad():
            _jacobian_input, _jacobians_h = torch.autograd.functional.jacobian(
                self.adapter.step, (self.input, self.state)
            )

        # Preserve matrix dimensions even for a single input or state feature.
        return _jacobians_h, _jacobian_input

    def _eigendecomposition(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute eigenvalues and eigenvectors of the state Jacobian at zero input.

        Args:
            state: Reference state [D], or LSTM tuple (h, c) of [H] tensors.

        Returns:
            Real eigenvalue parts [D], imaginary parts [D], and complex
            eigenvectors [D, D] stored in columns, preserving device and
            precision. Strict local stability of a discrete-time fixed
            point requires every eigenvalue magnitude to be below one.
            Unit-magnitude eigenvalues are inconclusive at linear order.

        Note:
            For nonzero input, call jacobian(input, state) and apply
            torch.linalg.eig to the returned state Jacobian.
        """

        eigenvalues, eigenvectors = torch.linalg.eig(self.jac_rec)

        return eigenvalues.real, eigenvalues.imag, eigenvectors

    def _controllability_gramian(self) -> torch.Tensor:
        """Sum A^m B B.T (A.T)^m for at most ``gramian_k`` terms.

        Stop when the largest absolute entry of the change is below
        ``gramian_tol``. Store that absolute change as ``gramian_diff``
        (a float), including when the term limit is reached. With no
        terms, return a zero matrix and leave ``gramian_diff`` infinite.
        """
        gramian = torch.zeros_like(self.jac_rec)
        propagated_input = self.jac_inp
        self.gramian_diff = float("inf")
        for _ in range(self.gramian_k):
            next_gramian = gramian + propagated_input @ propagated_input.T
            self.gramian_diff = (next_gramian - gramian).abs().max().item()
            gramian = next_gramian
            if self.gramian_diff < self.gramian_tol:
                break
            propagated_input = self.jac_rec @ propagated_input
        return gramian
