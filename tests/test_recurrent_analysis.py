"""Check full-state dynamics across the architectures used by analysis.py."""

import importlib.util
from pathlib import Path

import pytest
import torch
from torch import nn

from rnntoolkit import FixedPointFinder, FlowFieldFinder, Linearization
from rnntoolkit.adapter import RNNAdapter


@pytest.fixture
def analysis(monkeypatch):
    examples = Path(__file__).resolve().parents[1] / "examples"
    monkeypatch.syspath_prepend(str(examples))
    spec = importlib.util.spec_from_file_location("example_analysis", examples / "analysis.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("name", ["RNNModel", "GRUModel", "LSTMModel"])
def test_collected_trajectories_match_sequence_model(analysis, name):
    torch.manual_seed(1)
    model = getattr(analysis, name)(2, 3, 2).double().eval()
    inputs = torch.randn(2, 5, 2, dtype=torch.double)
    predictions, states = analysis.collect_trajectories(model, inputs)
    initial = inputs.new_zeros(1, 2, 3)
    native = (initial, initial.clone()) if name == "LSTMModel" else initial
    hidden, final = model.rnn(inputs, native)
    torch.testing.assert_close(predictions, model.out_layer(hidden))
    torch.testing.assert_close(states[..., :3], hidden)
    if name == "LSTMModel":
        assert states.shape == (2, 5, 6)
        torch.testing.assert_close(states[:, -1, 3:], final[1].squeeze(0))
        # Compare every cell state to a native sequence prefix, not just the last.
        for t in range(inputs.shape[1]):
            _, (_, cell) = model.rnn(inputs[:, :t + 1], native)
            torch.testing.assert_close(states[:, t, 3:], cell.squeeze(0))


@pytest.mark.parametrize("cls", [nn.RNN, nn.GRU, nn.LSTM])
def test_linearization_matches_finite_differences(cls):
    torch.manual_seed(2)
    model = cls(1, 1, batch_first=True).double()
    adapter = RNNAdapter(model)
    lin = Linearization(model)
    state = torch.randn(adapter.state_size, dtype=torch.double)
    inp = torch.randn(1, dtype=torch.double)
    jac, jac_inp = lin.jacobian(inp, state)
    assert jac.shape == (adapter.state_size, adapter.state_size)
    assert jac_inp.shape == (adapter.state_size, 1)
    eps = 1e-6
    expected = torch.stack([
        (adapter(inp, state + eps * d) - adapter(inp, state - eps * d)) / (2 * eps)
        for d in torch.eye(adapter.state_size, dtype=torch.double)
    ], dim=1)
    torch.testing.assert_close(jac, expected, atol=1e-8, rtol=1e-5)
    expected_inp = (adapter(inp + eps, state) - adapter(inp - eps, state)) / (2 * eps)
    torch.testing.assert_close(jac_inp[:, 0], expected_inp, atol=1e-8, rtol=1e-5)
    delta_state = torch.full((3, adapter.state_size), 1e-5, dtype=torch.double)
    delta_input = torch.full_like(inp, 1e-5)
    approximate = lin(inp, state, delta_input, delta_state)
    actual = adapter((inp + delta_input).expand(3, -1), state + delta_state)
    torch.testing.assert_close(approximate, actual, atol=1e-8, rtol=1e-5)


@pytest.mark.parametrize("batch_first", [False, True])
def test_lstm_fixed_points_optimize_and_report_both_states(batch_first):
    model = nn.LSTM(1, 2, batch_first=batch_first)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
    # With zero weights, c' = c/2 and h' = tanh(c/2)/2: (0, 0) is the fixed point.
    initial = (torch.full((2, 2), 0.2), torch.full((2, 2), 0.4))
    finder = FixedPointFinder(
        model, lr_init=0.005, max_iters=500, tol_q=1e-12,
        do_exclude_distance_outliers=False,
    )
    _, fps = finder.find_fixed_points(initial, torch.zeros(1))
    assert fps.xstar.shape == (2, 4)
    assert fps.xstar.abs().max() < 1e-4
    torch.testing.assert_close(fps.x_init, torch.cat(initial, dim=-1))
    expected_next = RNNAdapter(model)(torch.zeros(2, 1), fps.xstar)
    torch.testing.assert_close(fps.F_xstar, expected_next)
    torch.testing.assert_close(fps.qstar, 0.5 * (fps.xstar - expected_next).square().sum(-1))


def test_lstm_flow_accepts_packed_states_and_tuples():
    torch.manual_seed(3)
    model = nn.LSTM(2, 2, batch_first=True)
    pair = (torch.randn(4, 2), torch.randn(4, 2))
    packed = RNNAdapter(model).pack_state(pair)
    inputs = torch.zeros(4, 2)
    finder = FlowFieldFinder(model, fit_states=pair, num_points=3)
    for method, extra in [(finder.find_nonlinear_flow, ()), (finder.find_linear_flow, (inputs,))]:
        packed_flows = method(packed, inputs, *extra)
        tuple_flows = method(pair, inputs, *extra)
        for expected, actual in zip(packed_flows, tuple_flows):
            torch.testing.assert_close(actual.x_vels, expected.x_vels)
            torch.testing.assert_close(actual.y_vels, expected.y_vels)
            assert torch.isfinite(actual.speeds).all()
