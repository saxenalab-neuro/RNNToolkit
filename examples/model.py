import torch.nn as nn


class RNNModel(nn.Module):
    """Single-layer, batch-first RNN with a linear readout at each timestep."""
    def __init__(self, inp_dim, hid_dim, out_dim):
        """Create the recurrent module and readout.

        Args:
            inp_dim: Number of input features.
            hid_dim: Hidden-state width H.
            out_dim: Number of prediction features.
        """
        super(RNNModel, self).__init__()
        self.rnn = nn.RNN(inp_dim, hid_dim, batch_first=True)
        self.out_layer = nn.Linear(hid_dim, out_dim)

    def forward(self, input, hx):
        """Run a batch of sequences and read out every hidden state.

        Args:
            input: Inputs [batch, time, inp_dim].
            hx: Initial hidden state [1, batch, hid_dim], or None for zeros.

        Returns:
            Predictions [batch, time, out_dim] and hidden trajectories
            [batch, time, hid_dim].
        """
        hx, _ = self.rnn(input, hx)
        out = self.out_layer(hx)
        return out, hx


class GRUModel(nn.Module):
    """Single-layer GRU with a linear readout at every timestep."""

    def __init__(self, inp_dim, hid_dim, out_dim):
        """Create a batch-first GRU and linear readout.

        Args:
            inp_dim: Number of input features.
            hid_dim: Hidden-state width H.
            out_dim: Number of prediction features.
        """
        super().__init__()
        self.rnn = nn.GRU(inp_dim, hid_dim, batch_first=True)
        self.out_layer = nn.Linear(hid_dim, out_dim)

    def forward(self, input, hx):
        """Run a batch of sequences and read out every hidden state.

        Args:
            input: Inputs [batch, time, inp_dim].
            hx: Initial hidden state [1, batch, hid_dim], or None for zeros.

        Returns:
            Predictions [batch, time, out_dim] and hidden trajectories
            [batch, time, hid_dim], matching RNNModel.
        """
        hidden, _ = self.rnn(input, hx)
        out = self.out_layer(hidden)
        return out, hidden


class LSTMModel(nn.Module):
    """Single-layer LSTM with a linear readout at every timestep."""

    def __init__(self, inp_dim, hid_dim, out_dim):
        """Create a batch-first LSTM and linear readout without projections.

        Args:
            inp_dim: Number of input features.
            hid_dim: Width H of both hidden and cell states.
            out_dim: Number of prediction features.
        """
        super().__init__()
        self.rnn = nn.LSTM(inp_dim, hid_dim, batch_first=True)
        self.out_layer = nn.Linear(hid_dim, out_dim)

    def forward(self, input, hx):
        """Run sequences from a native LSTM state and read out hidden states.

        Args:
            input: Inputs [batch, time, inp_dim].
            hx: Initial (h0, c0) tuple, each [1, batch, hid_dim], or None
                for zeros. A packed analysis tensor is not accepted here.

        Returns:
            Predictions [batch, time, out_dim] and hidden trajectories
            [batch, time, hid_dim]. Cell trajectories are not returned.
            Use analysis.collect_trajectories for full [h, c] analysis states.
        """
        hidden, _ = self.rnn(input, hx)
        out = self.out_layer(hidden)
        return out, hidden
