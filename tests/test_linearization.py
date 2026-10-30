import torch
import torch.nn as nn

from rnntoolkit import Linearization


def test_jacobian_matches_weighted_activation_grad():
    rnn = nn.RNN(input_size=2, hidden_size=2, batch_first=True, nonlinearity="relu")
    rnn.weight_hh_l0 = nn.Parameter(torch.tensor([[0.7, 0.5], [0.4, 0.3]]))
    rnn.weight_ih_l0 = nn.Parameter(torch.tensor([[1.0, 0.0], [0.0, -1.0]]))
    rnn.bias_hh_l0 = nn.Parameter(torch.tensor([0.0, 0.0]))
    rnn.bias_ih_l0 = nn.Parameter(torch.tensor([0.0, 0.0]))
    h = torch.tensor([0.2, -0.1])
    x = torch.tensor([1.0, 1.0])
    linearization = Linearization(rnn, x, h)

    jac, jac_inp = linearization.jac_rec, linearization.jac_inp

    expected_diag = torch.tensor([[1.0, 0.0], [0.0, 0.0]])
    expected_jac = expected_diag @ rnn.weight_hh_l0
    expected_inp = expected_diag @ rnn.weight_ih_l0

    assert torch.allclose(jac, expected_jac)
    assert torch.allclose(jac_inp, expected_inp)


def test_eigenvalues_match_expected():
    rnn = nn.RNN(input_size=2, hidden_size=2, batch_first=True, nonlinearity="relu")
    rnn.weight_hh_l0 = nn.Parameter(torch.tensor([[0.7, 0.5], [0.4, 0.3]]))
    rnn.weight_ih_l0 = nn.Parameter(torch.tensor([[1.0, 0.0], [0.0, 1.0]]))
    rnn.bias_hh_l0 = nn.Parameter(torch.tensor([0.0, 0.0]))
    rnn.bias_ih_l0 = nn.Parameter(torch.tensor([0.0, 0.0]))
    h = torch.tensor([0.2, -0.1])
    linearization = Linearization(rnn, torch.zeros(2), h)
    reals = linearization.jac_rec_real_eigs
    ims = linearization.jac_rec_im_eigs

    eigenvalues, _ = torch.linalg.eig(rnn.weight_hh_l0)

    # Split real and imaginary parts
    _reals = []
    for eigenvalue in eigenvalues:
        _reals.append(eigenvalue.real.item())
    _reals = torch.tensor(_reals)

    _ims = []
    for eigenvalue in eigenvalues:
        _ims.append(eigenvalue.imag.item())
    _ims = torch.tensor(_ims)

    assert torch.allclose(reals, _reals)
    assert torch.allclose(ims, _ims)


def test_eigendecomposition_uses_reference_input_and_preserves_precision():
    rnn = nn.RNN(
        input_size=3, hidden_size=2, batch_first=True, nonlinearity="relu"
    ).double()
    with torch.no_grad():
        rnn.weight_hh_l0.copy_(torch.diag(torch.tensor([0.5, 0.25])))
        rnn.weight_ih_l0.copy_(torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]))
        rnn.bias_hh_l0.zero_()
        rnn.bias_ih_l0.zero_()
    h = torch.tensor([0.2, -0.1], dtype=torch.float64)
    x = torch.tensor([-1.0, 1.0, 0.0], dtype=torch.float64)
    linearization = Linearization(rnn, x, h)

    # This input switches which ReLU unit is active compared with zero input.
    expected = torch.tensor([0.0, 0.25], dtype=torch.float64)
    torch.testing.assert_close(linearization.jac_rec_real_eigs, expected)
    torch.testing.assert_close(linearization.jac_rec_im_eigs, torch.zeros_like(expected))
    eigenvectors = linearization.jac_rec_eigvecs
    assert eigenvectors.dtype == torch.complex128
    assert eigenvectors.device == h.device
    eigenvalues = torch.complex(
        linearization.jac_rec_real_eigs, linearization.jac_rec_im_eigs
    )
    torch.testing.assert_close(
        linearization.jac_rec.to(eigenvectors.dtype) @ eigenvectors,
        eigenvectors * eigenvalues,
    )
