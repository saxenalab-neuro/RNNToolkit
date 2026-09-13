import torch
import torch.nn as nn


class Linearization:
    def __init__(
        self,
        rnn: nn.RNN,
    ):
        """
        Linearization object that stores methods for local analyses of mRNNs

        This class is not meant for subclassing, for those subclassing flow fields
        and fixed points make your own linearization object

        Args:
            mrnn: mRNN object
        """
        self.rnn = rnn

    def __call__(
        self,
        input: torch.Tensor,
        h: torch.Tensor,
        delta_input: torch.Tensor,
        delta_h: torch.Tensor,
    ) -> torch.Tensor:
        """
        Calling foward method to apply affine function at given state and input
        """
        return self.forward(input, h, delta_input, delta_h)

    def forward(
        self,
        input: torch.Tensor,
        h: torch.Tensor,
        delta_input: torch.Tensor,
        delta_h: torch.Tensor,
    ) -> torch.Tensor:
        """
        First order taylor exansion of RNN at a given point and input

        Args:
            input (Tensor): 1D tensor of input for network at a given state
            h (Tensor): 1D tensor of network state to linearize about
            delta_input (Tensor): 1D tensor for perturbation of input to apply to input jacobian
            delta_h (Tensor): batched perturbations of state
        """

        # Assert correct shapes
        assert input.dim() == 1
        assert h.dim() == 1
        assert delta_input.dim() == 1

        # assert correct batch dimensions
        if delta_h.dim() > 1:
            delta_h = delta_h.flatten(start_dim=0, end_dim=-2)

        # Get jacobians
        _jacobian, _jacobian_inp = self.jacobian(input, h)

        # reshape to pass into RNN
        input = input.unsqueeze(0).unsqueeze(0)
        h = h.unsqueeze(0).unsqueeze(0)

        with torch.no_grad():
            # Get h_next for affine function
            _, h_next = self.rnn(input, h)

        # If there is only a single input there becomes a shape issue with squeezing
        if delta_input.shape == (1,) or _jacobian_inp.dim() == 1:
            _jacobian_inp = _jacobian_inp.unsqueeze(1)

        h_pert = (
            h_next.squeeze(0)
            + (_jacobian @ delta_h.T).T
            + (_jacobian_inp @ delta_input)
        )

        return h_pert

    def jacobian(
        self, input: torch.Tensor, h: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Linearize the dynamics around a state and return the Jacobian.

        Computes the Jacobian of the RNN update with respect to the hidden state
        evaluated at the provided state ``h`` and input ``input``.

        Args:
            input (torch.Tensor): 1D tensor representing the input for the network at state h
            h (torch.Tensor): 1D tensor representing the desired state h for taylor expansion

        Returns:
            _jacobian_h (torch.Tensor): jacobian of h_t+1 w respect to h_t
            _jacobian_inp (torch.Tensor): jacobian of h_t+1 w respect to input_t
        """
        assert h.dim() == 1
        assert input.dim() == 1

        input = input.unsqueeze(0).unsqueeze(0)
        h = h.unsqueeze(0).unsqueeze(0)

        with torch.no_grad():
            _, _jacobians_h = torch.autograd.functional.jacobian(self.rnn, (input, h))

        _jacobian_input, _jacobian_h = _jacobians_h

        return _jacobian_h.squeeze(), _jacobian_input.squeeze()

    def eigendecomposition(
        self, h: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Linearize the network and compute eigen decomposition.

        The recurrent Jacobian is evaluated at a zero input with the RNN input width.

        Args:
            h (torch.Tensor): 1D hidden state at which to evaluate the recurrent Jacobian

        Returns:
            torch.Tensor: Real parts of eigenvalues.
            torch.Tensor: Imag parts of eigenvalues.
            torch.Tensor: Eigenvectors stacked column-wise.
        """
        assert h.dim() == 1

        zero_input = h.new_zeros(self.rnn.input_size)
        _jacobian, _ = self.jacobian(zero_input, h)
        eigenvalues, eigenvectors = torch.linalg.eig(_jacobian)

        # Split real and imaginary parts
        reals = []
        for eigenvalue in eigenvalues:
            reals.append(eigenvalue.real.item())
        reals = torch.tensor(reals)

        ims = []
        for eigenvalue in eigenvalues:
            ims.append(eigenvalue.imag.item())
        ims = torch.tensor(ims)

        return reals, ims, eigenvectors
