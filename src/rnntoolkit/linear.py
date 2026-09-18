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
    ):
        """Initialize the differentiable one-step adapter.

        Args:
            rnn: Single-layer, unidirectional nn.RNN, nn.GRU, or nn.LSTM.
                Projected LSTMs are not supported. Tensors must match the
                module's device and dtype.
        """
        self.rnn = rnn
        self.adapter = RNNAdapter(self.rnn)

    def __call__(
        self,
        input: torch.Tensor,
        state: torch.Tensor | Tuple,
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
        return self.forward(input, state, delta_input, delta_h)

    def forward(
        self,
        input: torch.Tensor,
        state: torch.Tensor | Tuple,
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
        if not self.adapter.is_packed(state):
            state = self.adapter.pack_state(state)

        if not self.adapter.is_packed(delta_state):
            delta_state = self.adapter.pack_state(delta_state)

        # Assert correct shapes
        assert delta_input.dim() == 1

        # assert correct batch dimensions
        if delta_state.dim() > 1:
            delta_state = delta_state.flatten(start_dim=0, end_dim=-2)

        # Get jacobians
        _jacobian, _jacobian_inp = self.jacobian(input, state)

        with torch.no_grad():
            # Get h_next for affine function
            next_state = self.adapter(input, state)

        h_pert = (
            next_state
            + delta_state @ _jacobian.T
            + (_jacobian_inp @ delta_input)
        )

        return h_pert

    def jacobian(
        self, input: torch.Tensor, state: torch.Tensor | Tuple
    ) -> tuple[torch.Tensor, torch.Tensor]:
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

        if not self.adapter.is_packed(state):
            state = self.adapter.pack_state(state)

        assert state.dim() == 1
        assert input.dim() == 1

        with torch.no_grad():
            _jacobian_input, _jacobians_h = torch.autograd.functional.jacobian(self.adapter.step, (input, state))

        # Preserve matrix dimensions even for a single input or state feature.
        return _jacobians_h, _jacobian_input

    def eigendecomposition(
        self, state: torch.Tensor | Tuple
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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

        if not self.adapter.is_packed(state):
            state = self.adapter.pack_state(state)

        zero_input = state.new_zeros(self.rnn.input_size)
        _jacobian, _ = self.jacobian(zero_input, state)
        eigenvalues, eigenvectors = torch.linalg.eig(_jacobian)

        return eigenvalues.real, eigenvalues.imag, eigenvectors
