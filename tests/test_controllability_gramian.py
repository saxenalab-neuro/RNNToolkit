import pytest
import torch
from torch import nn

from rnntoolkit import Linearization


def make_linearization(a, b, k, tol):
    rnn = nn.RNN(b.shape[1], a.shape[0], nonlinearity="relu").double()
    with torch.no_grad():
        rnn.weight_hh_l0.copy_(a)
        rnn.weight_ih_l0.copy_(b)
        rnn.bias_hh_l0.fill_(1)
        rnn.bias_ih_l0.zero_()
    return Linearization(
        rnn, torch.zeros(b.shape[1], dtype=torch.float64),
        torch.zeros(a.shape[0], dtype=torch.float64),
        gramian_k=k, gramian_tol=tol,
    )


def test_gramian_stops_at_tolerance_and_records_change():
    linearization = make_linearization(torch.tensor([[0.5]]), torch.ones(1, 1), 20, 0.02)

    # Successive terms are 1, 1/4, 1/16, 1/64; the fourth is below tolerance.
    torch.testing.assert_close(linearization.ctrl_gramian, torch.tensor([[1.328125]], dtype=torch.float64))
    assert linearization.gramian_diff == pytest.approx(0.015625)


def test_gramian_respects_limit_and_uses_matrix_powers():
    a = torch.tensor([[0.5, 0.2], [0.0, 0.3]], dtype=torch.float64)
    b = torch.tensor([[1.0], [-2.0]], dtype=torch.float64)
    linearization = make_linearization(a, b, 3, 0.0)
    terms = [torch.linalg.matrix_power(a, m) @ b for m in range(3)]
    expected = sum(term @ term.T for term in terms)

    torch.testing.assert_close(linearization.ctrl_gramian, expected)
    assert linearization.gramian_diff == pytest.approx((terms[-1] @ terms[-1].T).abs().max().item())


def test_gramian_with_no_terms():
    linearization = make_linearization(torch.eye(2), torch.ones(2, 1), 0, 0.01)

    torch.testing.assert_close(linearization.ctrl_gramian, torch.zeros(2, 2, dtype=torch.float64))
    assert linearization.gramian_diff == float("inf")
