import torch
import torch.nn as nn
import pytest
from sklearn.decomposition import PCA
from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_is_fitted

from rnntoolkit import FlowFieldFinder


def test_find_nonlinear_flow_returns_flow_fields():
    rnn = nn.RNN(input_size=2, hidden_size=2, batch_first=True, nonlinearity="tanh")

    states = torch.tensor([[0.1, 0.2], [0.2, 0.1], [0.3, -0.1]])
    inp = torch.tensor([[0.0, 0.0], [0.1, 0.2], [-0.1, 0.1]])

    finder = FlowFieldFinder(
        rnn,
        num_points=3,
        x_offset=1,
        y_offset=1,
        x_center=0,
        y_center=0,
        fit_states=states,
    )

    flows = finder.find_nonlinear_flow(states, inp)

    assert len(flows) == states.shape[0]
    assert flows[0].x_vels.shape == (3, 3)
    assert flows[0].grid.shape == (3, 3, 2)


def test_inverse_grid_shapes_match_num_points():
    rnn = nn.RNN(input_size=2, hidden_size=2, batch_first=True, nonlinearity="tanh")
    traj = torch.tensor([[0.0, 0.1], [0.2, 0.3]])
    finder = FlowFieldFinder(
        rnn,
        num_points=4,
        x_offset=1,
        y_offset=1,
        x_center=0,
        y_center=0,
        fit_states=traj,
    )

    low_dim_grid, inv_grid = finder._inverse_grid(-1, 1, -1, 1)

    assert low_dim_grid.shape == (16, 2)
    assert inv_grid.shape == (16, 2)


@pytest.mark.parametrize("batched", [False, True])
def test_transforms_use_axes_when_given(batched):
    rnn = nn.RNN(input_size=2, hidden_size=3, batch_first=True, nonlinearity="tanh")
    fit_states = torch.tensor([[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]])
    if batched:
        fit_states = fit_states.unsqueeze(0)
    axes = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    states = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    low_dim_states = torch.tensor([[7.0, 8.0], [9.0, 10.0]])

    finder = FlowFieldFinder(
        rnn,
        num_points=3,
        x_offset=1,
        y_offset=1,
        fit_states=fit_states,
        axes=axes,
    )

    mean = torch.full((3,), 10.0 / 3)
    assert torch.allclose(finder.fit_states_mean, mean)
    assert torch.allclose(finder.transform(states), (states - mean)[:, :2])
    assert torch.allclose(finder.transform(mean.unsqueeze(0)), torch.zeros(1, 2))
    assert torch.allclose(
        finder.inverse_transform(low_dim_states),
        torch.tensor([[7.0, 8.0, 0.0], [9.0, 10.0, 0.0]]) + mean,
    )


def test_transforms_use_pca_fit_states_without_axes():
    rnn = nn.RNN(input_size=2, hidden_size=3, batch_first=True, nonlinearity="tanh")
    fit_states = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.5],
            [0.0, 1.0, 1.5],
            [1.0, 1.0, 2.0],
        ]
    )
    states = torch.tensor([[0.5, 0.0, 0.25], [0.25, 0.75, 1.25]])
    low_dim_states = torch.tensor([[0.1, 0.2], [0.3, 0.4]])
    expected_pca = PCA(n_components=2).fit(fit_states)

    finder = FlowFieldFinder(
        rnn, num_points=3, x_offset=1, y_offset=1, fit_states=fit_states
    )

    assert torch.allclose(
        torch.as_tensor(finder.transform(states), dtype=states.dtype),
        torch.as_tensor(expected_pca.transform(states), dtype=states.dtype),
    )
    assert torch.allclose(
        torch.as_tensor(finder.inverse_transform(low_dim_states), dtype=states.dtype),
        torch.as_tensor(
            expected_pca.inverse_transform(low_dim_states), dtype=states.dtype
        ),
    )


def test_find_nonlinear_flow_fits_pca_from_states_when_no_fit_states_or_axes():
    rnn = nn.RNN(input_size=2, hidden_size=3, batch_first=True, nonlinearity="tanh")
    states = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [0.5, 0.0, 0.25],
            [0.0, 0.5, 0.75],
            [0.5, 0.5, 1.0],
        ]
    )
    inp = torch.zeros(states.shape[0], 2)
    finder = FlowFieldFinder(
        rnn,
        num_points=3,
        x_offset=1,
        y_offset=1,
        fit_states=None,
    )

    with pytest.raises(NotFittedError):
        check_is_fitted(finder.reduce_obj)

    finder.find_nonlinear_flow(states, inp)

    assert torch.allclose(
        torch.as_tensor(finder.reduce_obj.mean_, dtype=states.dtype), states.mean(dim=0)
    )


def test_find_linear_flow_fits_pca_from_states_when_no_fit_states_or_axes():
    rnn = nn.RNN(input_size=2, hidden_size=3, batch_first=True, nonlinearity="tanh")
    states = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [0.5, 0.0, 0.25],
            [0.0, 0.5, 0.75],
            [0.5, 0.5, 1.0],
        ]
    )
    inp = torch.zeros(states.shape[0], 2)
    delta_inp = torch.zeros_like(inp)
    finder = FlowFieldFinder(
        rnn,
        num_points=3,
        x_offset=1,
        y_offset=1,
        fit_states=None,
    )

    with pytest.raises(NotFittedError):
        check_is_fitted(finder.reduce_obj)

    finder.find_linear_flow(states, inp, delta_inp)

    assert torch.allclose(
        torch.as_tensor(finder.reduce_obj.mean_, dtype=states.dtype), states.mean(dim=0)
    )
