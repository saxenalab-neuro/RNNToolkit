"""Convert standard recurrent modules into flat-state, one-step dynamics."""

import torch
from torch import nn


class RNNAdapter:
    """Expose an RNN, GRU, or LSTM as a differentiable state transition.

    Analysis states are ordinary tensors: ``h`` for RNNs/GRUs and
    ``[h, c]`` concatenated along the last dimension for LSTMs. Including the
    cell state makes the LSTM state sufficient to determine its next update.

    Only single-layer, unidirectional modules are supported. Projected LSTMs
    are excluded so hidden and cell states have the same width. This adapter
    wraps the supplied module by reference; it does not copy weights, change
    training mode, freeze parameters, or move tensors between devices/dtypes.
    Inputs and states must already match the module's device and dtype.

    ``step`` (also available through calling the adapter) accepts either an
    unbatched input/state pair of shapes ``[input_size]`` and ``[state_size]``,
    or a batched pair of shapes ``[B, input_size]`` and ``[B, state_size]``.
    It returns the next state with the same shape as the supplied state.
    Broadcasting and sequence processing are intentionally left to callers.

    Packing, unpacking, and stepping preserve autograd connections, allowing
    optimization over states and Jacobians with respect to inputs or states.

    Args:
        rnn: A ``torch.nn.RNN``, ``torch.nn.GRU``, or ``torch.nn.LSTM``.

    Attributes:
        input_size: Number of input features.
        hidden_size: Number of features in each hidden/cell state.
        state_size: Width of the flat analysis state (twice hidden_size for
            LSTMs).

    Example:
        >>> model = nn.LSTM(2, 3, batch_first=True)
        >>> adapter = RNNAdapter(model)
        >>> state = adapter.pack_state((torch.zeros(4, 3), torch.zeros(4, 3)))
        >>> next_state = adapter(torch.zeros(4, 2), state)
        >>> h_next, c_next = adapter.unpack_state(next_state)
        >>> h_next.shape
        torch.Size([4, 3])
    """

    def __init__(self, rnn: nn.RNN | nn.GRU | nn.LSTM):
        """Store a supported recurrent module and its state dimensions.

        Args:
            rnn: Single-layer, unidirectional nn.RNN, nn.GRU, or nn.LSTM.

        Raises:
            TypeError: If rnn is not a supported PyTorch recurrent module.
            ValueError: For multiple layers, bidirectionality, or LSTM projections.
        """
        if not isinstance(rnn, (nn.RNN, nn.GRU, nn.LSTM)):
            raise TypeError("rnn must be an nn.RNN, nn.GRU, or nn.LSTM module")
        if rnn.num_layers != 1 or rnn.bidirectional:
            raise ValueError("Only single-layer, unidirectional modules are supported")
        if isinstance(rnn, nn.LSTM) and rnn.proj_size != 0:
            raise ValueError("Projected LSTMs are not supported")

        self.rnn = rnn
        self.is_lstm = isinstance(rnn, nn.LSTM)
        self.input_size = rnn.input_size
        self.hidden_size = rnn.hidden_size
        self.state_size = self.hidden_size * (2 if self.is_lstm else 1)

    @staticmethod
    def _check_width(tensor: torch.Tensor, width: int, name: str) -> None:
        """Require a tensor with at least one dimension and the specified width.

        Raises:
            TypeError: If tensor is not a torch.Tensor.
            ValueError: If tensor is scalar or its last dimension differs from width.
        """
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")
        if tensor.ndim < 1 or tensor.shape[-1] != width:
            raise ValueError(f"{name} must have last dimension {width}")

    def is_packed(
        self, state: torch.Tensor | tuple[torch.Tensor, torch.Tensor]
    ) -> bool:
        """Return whether a state has the packed analysis representation.

        For LSTMs, a packed state is a tensor with at least one dimension
        and last dimension ``state_size`` (concatenated ``[h, c]``). Tuples
        and tensors of other shapes return False. Leading dimensions may
        represent batches or trajectories.

        For RNNs and GRUs, always return True because no packing is needed.
        This is a representation check, not full state validation: False
        does not guarantee a valid unpacked LSTM pair, and True does not
        check device, dtype, or compatibility with ``step``'s batch shape.
        """
        if not self.is_lstm:
            return True
        return (
            isinstance(state, torch.Tensor)
            and state.ndim >= 1
            and state.shape[-1] == self.state_size
        )

    def pack_state(
        self, state: torch.Tensor | tuple[torch.Tensor, torch.Tensor]
    ) -> torch.Tensor:
        """Pack ``h`` or an LSTM ``(h, c)`` pair along the last dimension.

        Leading dimensions are preserved, making this usable with individual
        states, batches, or recorded trajectories. LSTM components must have
        identical shapes, devices, and dtypes. Already-packed LSTM tensors are not accepted; use is_packed first.
        This method does not remove a PyTorch layer dimension; remove it before passing a batch to ``step``.
        """
        if not self.is_lstm:
            self._check_width(state, self.hidden_size, "state")
            return state
        if not isinstance(state, tuple) or len(state) != 2:
            raise TypeError("LSTM state must be an (h, c) tuple")
        h, c = state
        self._check_width(h, self.hidden_size, "h")
        self._check_width(c, self.hidden_size, "c")
        if h.shape != c.shape or h.device != c.device or h.dtype != c.dtype:
            raise ValueError("h and c must have identical shapes, devices, and dtypes")
        return torch.cat((h, c), dim=-1)

    def unpack_state(
        self, state: torch.Tensor
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Return native state components while preserving leading dimensions.

        Args:
            state: Packed tensor [..., state_size]. LSTM feature order is [h, c].

        Returns:
            The original tensor for RNN/GRU; an (h, c) tuple of views with
            shape [..., hidden_size] for LSTM. No layer dimension is added.
        """
        self._check_width(state, self.state_size, "state")
        if self.is_lstm:
            return state[..., : self.hidden_size], state[..., self.hidden_size :]
        return state

    def step(self, input: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        """Advance a flat state by one timestep using the wrapped module.

        Input and state must both be 1D or both be 2D with the same batch
        size. The module's ``batch_first`` setting is handled internally;
        the adapter's own batched interface always places batch first.

        Args:
            input: Tensor [I] or [B, I].
            state: Packed tensor [D] or [B, D]; no native LSTM tuples.

        Returns:
            Differentiable packed next state with the same shape as state.
        """
        self._check_width(input, self.input_size, "input")
        self._check_width(state, self.state_size, "state")
        if state.ndim not in (1, 2) or input.ndim != state.ndim:
            raise ValueError("input and state must both be 1D or both be 2D")
        if input.shape[:-1] != state.shape[:-1]:
            raise ValueError("input and state must have matching batch sizes")

        unbatched = state.ndim == 1
        if unbatched:
            input, state = input.unsqueeze(0), state.unsqueeze(0)

        native_state = self.unpack_state(state)
        if self.is_lstm:
            h, c = native_state
            native_state = (h.unsqueeze(0), c.unsqueeze(0))
        else:
            native_state = native_state.unsqueeze(0)

        time_dim = 1 if self.rnn.batch_first else 0
        _, next_native = self.rnn(input.unsqueeze(time_dim), native_state)
        next_state = self.pack_state(next_native).squeeze(0)
        return next_state.squeeze(0) if unbatched else next_state

    def __call__(self, input: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        """Advance one timestep; alias for :meth:`step`.

        Args:
            input: Input [I] or [B, I].
            state: Packed state [D] or [B, D], matching input batch dimensions.

        Returns:
            Packed next state with the same shape as state. LSTM tuples must
            be packed before calling; this alias does not auto-pack them.
        """
        return self.step(input, state)
