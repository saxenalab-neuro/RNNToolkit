# RNNToolkit

[![Documentation](https://img.shields.io/badge/docs-Read%20the%20Docs-blue)](https://rnntoolkit.readthedocs.io/en/latest/)

RNNToolkit provides PyTorch tools for fixed-point discovery, local linearization, two-dimensional flow fields, and interactive visualization of recurrent networks.

## Supported networks and states

The concrete analysis tools support **single-layer, unidirectional `torch.nn.RNN`, `torch.nn.GRU`, and `torch.nn.LSTM`**, with either `batch_first` setting. Projected LSTMs and cell modules such as `LSTMCell` are not supported by `RNNAdapter`.

| Network | Packed analysis state | State width D |
| --- | --- | --- |
| RNN / GRU | Hidden state `h` | H |
| LSTM | Concatenated `[h, c]` | 2H |

LSTM analysis includes both hidden and cell state: a stationary `h` alone does not establish a fixed point. The main analysis methods accept either packed tensors or `(h, c)` tuples with matching component shapes:

- `Linearization.jacobian`, `eigendecomposition`, and `forward`/calling the object (including state perturbations).
- `FixedPointFinder.find_fixed_points`.
- `FlowFieldFinder.find_nonlinear_flow`, `find_linear_flow`, and constructor `fit_states`.

State outputs are packed. **Sampling, projection helpers, plotting, and visualizer trajectory inputs require packed tensors.** `adapter.step()` also requires packed states. `adapter.pack_state()` expects an unpacked LSTM tuple; it does not accept an already-packed LSTM tensor.

## Installation

Use Python 3.11+ for the current source tree (`typing.Self` is used). Install runtime dependencies and the editable package from the repository root:

```bash
pip install torch numpy scikit-learn matplotlib pygame
pip install -e .
```

The current package metadata does not declare all runtime dependencies. The examples below use float32 CPU tensors; PCA uses scikit-learn and needs detached CPU data.

## Quick start

### Linearization with a GRU

```python
import torch
from rnntoolkit import Linearization

rnn = torch.nn.GRU(input_size=2, hidden_size=4, batch_first=True)
lin = Linearization(rnn)
state = torch.zeros(4)
input = torch.zeros(2)
J_state, J_input = lin.jacobian(input, state)  # [4, 4], [4, 2]
real, imag, eigenvectors = lin.eigendecomposition(state)  # zero input
```

`forward(input, state, delta_input, delta_state)` returns the affine approximation to the next state. For LSTMs, the Jacobian is `[2H, 2H]`. Strict local stability at a discrete-time fixed point requires all eigenvalue magnitudes to be below one; real parts alone are insufficient.

### LSTM state packing and fixed points

```python
from rnntoolkit import FixedPointFinder
from rnntoolkit.adapter import RNNAdapter

lstm = torch.nn.LSTM(input_size=2, hidden_size=4, batch_first=True)
adapter = RNNAdapter(lstm)
h = torch.randn(16, 4)
c = torch.randn(16, 4)
packed = adapter.pack_state((h, c))  # [16, 8]
assert adapter.is_packed(packed)
h_again, c_again = adapter.unpack_state(packed)

finder = FixedPointFinder(lstm, max_iters=100)
unique_fps, all_fps = finder.find_fixed_points((h, c), torch.zeros(2))
# finder.find_fixed_points(packed, torch.zeros(2)) also works.
# unique_fps.xstar contains packed [h, c] states.
```

The objective is `q = 0.5 * ||F(input, state) - state||²`, including both LSTM components. Inspect `qstar` before treating a candidate as a fixed point; a short optimization need not converge. The finder currently disables model parameter gradients without restoring their flags. Jacobians and eigenvalues are computed separately with `Linearization`.

### LSTM flow fields

```python
from rnntoolkit import FlowFieldFinder

flow_finder = FlowFieldFinder(lstm, fit_states=(h, c), num_points=25)
inputs = torch.zeros(16, 2)
nonlinear = flow_finder.find_nonlinear_flow(packed, inputs)
linear = flow_finder.find_linear_flow((h, c), inputs, torch.zeros_like(inputs))
```

PCA is fitted on the full packed state. Explicit flow axes must have shape `[2, D]`, hence `[2, 2H]` for LSTMs. Fields show projected one-step displacements; their normalized speeds are not an energy function or the full-state residual. A two-dimensional plot does not establish full-state stability.

## Run the example analyses

```bash
python examples/analysis.py
```

This runs the RNN, GRU, and LSTM checkpoints in `examples/`, saves six figures per architecture under `examples/analysis_results/{rnn,gru,lstm}/`, and allows up to 35,000 fixed-point iterations per architecture. There are no command-line options. To change the iteration budget, edit `main()` or call `run_analysis(..., max_iters=..., model_class=...)` from Python with the matching model class.

`examples/model.py` contains `RNNModel`, `GRUModel`, and `LSTMModel`. Their forward methods return predictions and hidden trajectories; LSTM forward accepts native `(h0, c0)` and does not return cell trajectories. `analysis.collect_trajectories` records full `[h, c]` at every timestep for the analyses, while predictions still use only `h`.

## Package layout

| Component | Purpose |
| --- | --- |
| `rnntoolkit.adapter` | State packing and differentiable one-step updates |
| `rnntoolkit.fixed_points` | Fixed-point optimization, sampling, and collections |
| `rnntoolkit.linear` | Jacobians, affine approximations, and eigenmodes |
| `rnntoolkit.flow_fields` | Flow fields in a two-dimensional projection |
| `rnntoolkit.flow_visualizer` | Interactive flow-field viewer |
