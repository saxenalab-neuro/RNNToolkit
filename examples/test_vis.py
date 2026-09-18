"""Visualize simple RNN, GRU, and LSTM memory dynamics.

Run with --model rnn (default), gru, lstm, or all. Close each window to
advance when running all three. The ReLU RNN has an exact line attractor
on its nonnegative first coordinate; the gated examples demonstrate
approximate integration/slow memory with contracting transverse directions.
"""

import argparse

import numpy as np
import pygame
import torch
import torch.nn as nn

from rnntoolkit import FlowFieldVisualizer
from rnntoolkit.adapter import RNNAdapter


def make_line_attractor():
    """Create h[t+1] = ReLU(W_hh h[t] + W_ih u[t]).

    With W_hh[0, 0] = 1, the first neuron is the line-attractor/integrator
    coordinate. The other two neurons contract toward zero.
    """
    rnn = nn.RNN(input_size=1, hidden_size=3, batch_first=True, nonlinearity="relu")
    with torch.no_grad():
        rnn.weight_ih_l0.zero_()
        rnn.weight_ih_l0[0, 0] = 1.0
        rnn.weight_hh_l0.zero_()
        rnn.weight_hh_l0[0, 0] = 1.0
        rnn.weight_hh_l0[1, 1] = 0.8
        rnn.weight_hh_l0[2, 2] = 0.7
        rnn.bias_ih_l0.zero_()
        rnn.bias_hh_l0.zero_()
    return rnn


def make_gru_line_attractor():
    """Create a GRU with a locally neutral first coordinate at zero input.

    Constant reset/update gates r = z = 0.5 give
    h' = 0.5 h + 0.5 tanh(0.5 W_hn h + W_in u).
    At the origin the state Jacobian has eigenvalues 1, 0.8, and 0.65.
    The first coordinate approximates integration near zero; tanh makes
    it contract away from zero, so this is not an exact line attractor.
    """
    gru = nn.GRU(input_size=1, hidden_size=3, batch_first=True)
    with torch.no_grad():
        for parameter in gru.parameters():
            parameter.zero_()
        # PyTorch stacks GRU gates in reset, update, candidate order.
        gru.weight_hh_l0[6:9].copy_(torch.diag(torch.tensor([2.0, 1.2, 0.6])))
        gru.weight_ih_l0[6, 0] = 2.0
    return gru


def make_lstm_line_attractor():
    """Create an LSTM with a slowly decaying first cell-memory coordinate.

    Forget gates are fixed at 0.999, 0.8, and 0.6; input/output gates are
    0.5. Thus c' = f*c + 0.5*tanh(W_ig*u), and h' = 0.5*tanh(c').
    A constant input drives only the first cell coordinate. At zero input
    this is a leaky memory, not an exact line attractor. Its full-state
    eigenvalues are the three forget factors and three zeros.
    """
    lstm = nn.LSTM(input_size=1, hidden_size=3, batch_first=True)
    with torch.no_grad():
        for parameter in lstm.parameters():
            parameter.zero_()
        # PyTorch stacks LSTM gates in input, forget, candidate, output order.
        lstm.bias_ih_l0[3:6].copy_(torch.logit(torch.tensor([0.999, 0.8, 0.6])))
        lstm.weight_ih_l0[6, 0] = 2.0
    return lstm


@torch.no_grad()
def collect_states(rnn, inputs, initial_state):
    """Roll out inputs [B, T, I] from packed initial states [B, D].

    Returns complete post-update trajectories [B, T, D]. For an LSTM,
    D = 2H and every timestep contains [h, c], including its cell memory.
    """
    adapter = RNNAdapter(rnn)
    state = initial_state
    trajectory = []
    for timestep_input in inputs.unbind(dim=1):
        state = adapter(timestep_input, state)
        trajectory.append(state)
    return torch.stack(trajectory, dim=1)


def visualize_line_attractor(model_type="rnn"):
    """Open a flow-field viewer for the selected rnn, gru, or lstm example.

    Three trials receive a constant input with different starting states.
    PCA fits complete trajectories: width three for RNN/GRU, six for LSTM.
    The LSTM readout and cell coordinates are both retained in its flow map.
    """
    factories = {
        "rnn": make_line_attractor,
        "gru": make_gru_line_attractor,
        "lstm": make_lstm_line_attractor,
    }
    rnn = factories[model_type]()
    rnn.eval()

    # Three trials share the same drive to the first memory coordinate.
    n_trials = 3
    n_time = 150
    static_input = 0.02
    inputs = torch.full((n_trials, n_time, 1), static_input)
    initial = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.4, 0.1], [2.0, 0.8, 0.2]])
    if model_type == "gru":
        # Start near the origin where the first coordinate is approximately neutral.
        initial = torch.tensor([[-0.2, 0.0, 0.0], [0.0, 0.2, 0.1], [0.2, 0.4, 0.2]])
    elif model_type == "lstm":
        # Start on h = o*tanh(c), consistent with the fixed output gate.
        initial = RNNAdapter(rnn).pack_state((0.5 * torch.tanh(initial), initial))

    states = collect_states(rnn, inputs, initial)

    print(f"Visualizing {model_type.upper()}")
    print("inputs:", inputs.shape)
    print("packed state trajectories:", states.shape)
    print("final state:", states[0, -1])

    # Fit PCA on the full state, including both h and c for LSTMs.
    # Closing a viewer calls pygame.quit(); reinitialize for --model all.
    pygame.init()
    visualizer = FlowFieldVisualizer(
        rnn,
        fit_states=states,
        num_points=15,
        x_offset=3,
        y_offset=3,
        x_center=1.5,
        y_center=0,
    )
    visualizer.visualize(inputs, states)


def main():
    """Select an architecture, or open all three demos sequentially."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("rnn", "gru", "lstm", "all"), default="rnn")
    args = parser.parse_args()
    torch.manual_seed(0)
    np.random.seed(0)
    model_types = ("rnn", "gru", "lstm") if args.model == "all" else (args.model,)
    for model_type in model_types:
        visualize_line_attractor(model_type)


if __name__ == "__main__":
    main()
