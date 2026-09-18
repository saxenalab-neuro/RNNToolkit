"""Run the flip-flop experiments from analysis.ipynb.

Loads the trained three-bit memory RNN, GRU, and LSTM, plots trial predictions, discovers
zero-input fixed points, plots nonlinear and linear flow fields, classifies
fixed-point stability, and measures alignment with unstable eigendirections.

Run from any directory with the project's dependencies installed::

    python /path/to/RNNToolkit/examples/analysis.py

The main function runs each architecture for up to 35,000 optimization
iterations and saves figures under analysis_results/<architecture>.
LSTM analyses use the full concatenated hidden and cell state [h, c].
"""

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA

# Resolve the local checkout and checkpoint independently of the working directory.
EXAMPLE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(EXAMPLE_DIR.parent / "src"))

from flip_flop_data import FlipFlopData
from model import RNNModel, GRUModel, LSTMModel
from rnntoolkit import FixedPointFinder, FlowFieldFinder, Linearization
from rnntoolkit.adapter import RNNAdapter


@torch.no_grad()
def collect_trajectories(model, inputs):
    """Collect predictions and complete post-update states at every timestep.

    Args:
        model: An example RNNModel, GRUModel, or LSTMModel with .rnn and
            .out_layer attributes. Evaluated without gradients; callers
            choose training/evaluation mode.
        inputs: Tensor [batch, time, input_size] matching the model device
            and dtype, with at least one timestep. Initial states are zero.

    Returns:
        Predictions [batch, time, out_dim] and packed states
        [batch, time, D], where D = H for RNN/GRU and D = 2H for LSTM.
        LSTM feature order is [h, c]; the readout uses only h. Stepping
        explicitly records cell states absent from sequence outputs.
    """
    adapter = RNNAdapter(model.rnn)
    state = inputs.new_zeros(inputs.shape[0], adapter.state_size)
    trajectories = []
    for timestep_input in inputs.unbind(dim=1):
        state = adapter(timestep_input, state)
        trajectories.append(state)
    trajectories = torch.stack(trajectories, dim=1)
    hidden = trajectories[..., :adapter.hidden_size]
    print(trajectories.shape)
    return model.out_layer(hidden), trajectories


def plot_fixed_points(states, trajectories, stability=None):
    """Plot packed fixed points and individual trials in three-dimensional PCA.

    Args:
        states: Fixed points [N, D]; packed [h, c] for LSTM.
        trajectories: Recorded states [batch, time, D] in the same format.
        stability: Optional boolean NumPy array [N] for stable-point colors.

    Returns:
        Matplotlib figure. PCA is fitted on fixed points when N >= 3,
        otherwise on trajectories. This fit is independent of flow-field PCA.
    """
    flat_trajectories = trajectories.reshape(-1, trajectories.shape[-1])
    # Preserve the notebook's fixed-point fit when enough points are available.
    fit_states = states if len(states) >= 3 else flat_trajectories
    pca = PCA(n_components=3).fit(fit_states)
    projected_states = pca.transform(states)
    projected_trials = pca.transform(flat_trajectories).reshape(
        *trajectories.shape[:-1], 3
    )
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    if stability is None:
        ax.scatter(*projected_states.T, color="blue", s=40, label="Fixed points")
    else:
        for label, mask, color in (
            ("Stable", stability, "blue"),
            ("Unstable or marginal", ~stability, "red"),
        ):
            if mask.any():
                ax.scatter(*projected_states[mask].T, color=color, s=40, label=label)
    for trial in projected_trials:
        ax.plot(*trial.T, color="black", alpha=0.4, linewidth=0.3)
    ax.set(title="Fixed points (PCA space)", xlabel="PC1", ylabel="PC2", zlabel="PC3")
    ax.legend()
    return fig


def plot_flow_fields(flows, projected_states, *, linear=False, stability=None):
    """Plot four nonlinear fields or nine linear fields with fixed-point markers.

    Args:
        flows: Sequence of FlowField objects to display.
        projected_states: Fixed-point coordinates [N, 2] from the same
            projection as flows. For linear plots, point i belongs to field i.
        linear: Select a 3-by-3 layout and one fixed-point marker per field.
        stability: Boolean array [N], required for linear plots.

    Returns:
        Matplotlib figure; unused panels are hidden.
    """
    side = 3 if linear else 2
    fig, axes = plt.subplots(side, side, figsize=(10, 10))
    for index, ax in enumerate(axes.flat):
        if index >= len(flows):
            ax.axis("off")
            continue
        flow = flows[index]
        ax.quiver(
            flow.grid[..., 0], flow.grid[..., 1], flow.x_vels, flow.y_vels,
            flow.speeds, cmap="viridis", alpha=0.85,
        )
        if linear:
            stable = stability[index]
            ax.scatter(
                *projected_states[index], color="blue" if stable else "red",
                marker="o" if stable else "x", s=50,
            )
            ax.set_title(f"Fixed point {index + 1}")
        else:
            ax.scatter(*projected_states.T, color="black", s=10)
            ax.set_title(f"Flow field t={index}")
        ax.set(xlabel="PC1", ylabel="PC2")
    kind = "Linear" if linear else "Nonlinear"
    fig.suptitle(f"{kind} flow fields + fixed points (PCA space)")
    fig.tight_layout()
    return fig


def analyze_stability(rnn, states):
    """Classify fixed points using the full zero-input state Jacobian.

    Args:
        rnn: Supported recurrent module.
        states: Packed fixed points [N, D], including [h, c] for LSTM.

    Returns:
        Boolean NumPy array [N] and a list of complex eigenvector matrices
        [D, K] for modes with magnitude greater than one. Strict stability
        requires all magnitudes below one. Unit-magnitude modes are marginal
        and are not included among the unstable eigenvectors.
    """
    linearization = Linearization(rnn)
    stability, unstable_vectors = [], []
    for state in states:
        real, imaginary, vectors = linearization.eigendecomposition(state)
        magnitudes = torch.sqrt(real.square() + imaginary.square())
        stability.append(bool(torch.all(magnitudes < 1)))
        unstable_vectors.append(vectors[:, magnitudes > 1])
    return np.asarray(stability), unstable_vectors


def plot_unstable_alignment(finder, states, unstable_vectors):
    """Plot mean perturbation angle to the nearest unstable mode.

    Args:
        finder: FlowFieldFinder whose projection and grid match states.
        states: Packed fixed points [N, D], including [h, c] for LSTM.
        unstable_vectors: List of complex [D, K] eigenvector matrices,
            with unstable modes stored in columns for each fixed point.

    Returns:
        Matplotlib figure showing mean angles for points with unstable modes.
        Zero-length perturbations are excluded. Eigenvector signs are
        arbitrary, so angles lie between zero and 90 degrees. For complex
        modes, alignment uses the real plane spanned by real/imaginary parts.
        Angles are measured in the full packed state with Euclidean geometry.
    """
    indices, mean_angles = [], []
    projected_states = finder._reduce_traj(states)
    for index, vectors in enumerate(unstable_vectors):
        if vectors.shape[1] == 0:
            continue
        bounds = finder._set_tv_bounds(projected_states[index])
        _, grid = finder._inverse_grid(*bounds)
        perturbations = grid - states[index]
        lengths = torch.linalg.vector_norm(perturbations, dim=1)
        valid = lengths > torch.finfo(perturbations.dtype).eps
        if not valid.any():
            continue
        directions = perturbations[valid] / lengths[valid, None]
        alignments = []
        for vector in vectors.T:
            plane = torch.stack((vector.real, vector.imag), dim=1)
            basis, singular_values, _ = torch.linalg.svd(plane, full_matrices=False)
            tolerance = max(plane.shape) * torch.finfo(plane.dtype).eps * singular_values[0]
            basis = basis[:, singular_values > tolerance]
            alignments.append(torch.linalg.vector_norm(directions @ basis, dim=1))
        closest = torch.stack(alignments).max(dim=0).values.clamp(0, 1)
        indices.append(index + 1)
        mean_angles.append(torch.rad2deg(torch.acos(closest)).mean().item())

    fig, ax = plt.subplots()
    ax.plot(indices, mean_angles, marker="o")
    ax.set(
        title="Alignment with unstable modes", xlabel="Fixed point",
        ylabel="Mean nearest-mode angle (degrees)", ylim=(0, 90),
    )
    if not indices:
        ax.text(0.5, 0.5, "No unstable modes found", transform=ax.transAxes, ha="center")
    return fig


def run_analysis(checkpoint, max_iters=35000, output_dir=None, *, model_class=RNNModel):
    """Run all six analysis stages for one trained three-bit memory model.

    Collects trajectories, finds zero-input fixed points, plots nonlinear
    flow, classifies stability, plots linear flow, and measures unstable-mode
    alignment. Every state-based stage uses full packed [h, c] for LSTMs.

    Args:
        checkpoint: Path to a state_dict matching model_class, with three
            input/output channels and 16 hidden units.
        max_iters: Maximum fixed-point optimization iterations (default 35000).
            A short run is useful for execution checks, not convergence claims.
        output_dir: Optional pathlib.Path for six PNG figures. Uses the Agg
            backend without interactive display; figures are closed even when
            output_dir is None.
        model_class: RNNModel (default), GRUModel, or LSTMModel. Must match
            the checkpoint architecture; filenames do not select the class.

    Returns:
        Pair (unique_fps, all_fps) with packed state tensors.

    Raises:
        RuntimeError: If no unique fixed points remain after filtering.
        FileNotFoundError: If the checkpoint is missing.
    """
    torch.manual_seed(0)
    np.random.seed(0)
    plt.switch_backend("Agg")
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)

    def finish_figure(fig, name):
        """Optionally save a named PNG, then close its Matplotlib figure."""
        if output_dir is not None:
            fig.savefig(output_dir / f"{name}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    n_bits, n_hidden = 3, 16
    n_trials, n_time = 64, 64
    model = model_class(n_bits, n_hidden, n_bits)
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    model.eval()
    print(model)

    data = FlipFlopData(n_bits=n_bits, n_time=n_time).generate_data(n_trials=n_trials)
    inputs = torch.from_numpy(data["inputs"])
    outputs, trajectories = collect_trajectories(model, inputs)
    print("inputs:", inputs.shape)
    print("outputs:", outputs.shape)
    print("analysis state trajectories:", trajectories.shape)
    finish_figure(FlipFlopData.plot_trials(data, outputs.numpy()), "01_trials")
    # The trial plot enables interactive mode; defer blocking display until the end.
    plt.ioff()

    print("Finding zero-input fixed points...")
    finder = FixedPointFinder(
        model.rnn, tol_unique=1e-1, tol_q=1e-10, tol_dq=1e-18, lr_init=0.01,
        max_iters=max_iters, verbose=True,
    )
    initial_states = finder.sample_states(trajectories, n_inits=128, noise_scale=0.25)
    unique_fps, all_fps = finder.find_fixed_points(initial_states, torch.zeros(n_bits))
    print("Unique fixed points:", unique_fps.n)
    if unique_fps.n == 0:
        raise RuntimeError("No fixed points remain after filtering; cannot analyze them.")
    states = unique_fps.xstar
    finish_figure(plot_fixed_points(states, trajectories), "02_fixed_points")

    print("Computing nonlinear flow fields...")
    nonlinear_finder = FlowFieldFinder(
        model.rnn, fit_states=trajectories, num_points=25,
        x_offset=2, y_offset=2, x_center=0, y_center=0, follow_traj=False,
    )
    zero_inputs = torch.zeros_like(inputs)
    nonlinear_flows = nonlinear_finder.find_nonlinear_flow(trajectories, zero_inputs)
    finish_figure(
        plot_flow_fields(nonlinear_flows[:4], nonlinear_finder.transform(states)),
        "03_nonlinear_flow",
    )

    print("Analyzing fixed-point eigenvalues...")
    stability, unstable_vectors = analyze_stability(model.rnn, states)
    print(f"Stable fixed points: {stability.sum()} / {len(states)}")
    finish_figure(plot_fixed_points(states, trajectories, stability), "04_stability")

    print("Computing linear flow fields around fixed points...")
    linear_finder = FlowFieldFinder(
        model.rnn, fit_states=states if len(states) >= 2 else trajectories,
        num_points=15, x_offset=0.5, y_offset=0.5,
        x_center=0, y_center=0, follow_traj=True,
    )
    fixed_point_inputs = torch.zeros(len(states), n_bits)
    linear_flows = linear_finder.find_linear_flow(
        states, fixed_point_inputs, torch.zeros_like(fixed_point_inputs)
    )
    finish_figure(
        plot_flow_fields(
            linear_flows[:9], linear_finder.transform(states),
            linear=True, stability=stability,
        ),
        "05_linear_flow",
    )
    finish_figure(
        plot_unstable_alignment(linear_finder, states, unstable_vectors),
        "06_unstable_alignment",
    )
    return unique_fps, all_fps


def main():

    """Analyze the RNN, GRU, and LSTM checkpoints sequentially.

    Reads flip_flop_rnn.pth, flip_flop_gru.pth, and flip_flop_lstm.pth
    beside this script. Saves six figures per model under
    analysis_results/rnn, analysis_results/gru, and analysis_results/lstm.
    Each run allows 35,000 fixed-point iterations. There are no CLI options.
    """
    max_iters = 50_000
    print("Performing RNN Analysis...")
    output_dir = EXAMPLE_DIR / "analysis_results/rnn"
    checkpoint = EXAMPLE_DIR / "flip_flop_rnn.pth"
    run_analysis(checkpoint, max_iters, output_dir, model_class=RNNModel)

    print("\nPerforming GRU Analysis...")
    output_dir = EXAMPLE_DIR / "analysis_results/gru"
    checkpoint = EXAMPLE_DIR / "flip_flop_gru.pth"
    run_analysis(checkpoint, max_iters, output_dir, model_class=GRUModel)

    print("\nPerforming LSTM Analysis...")
    output_dir = EXAMPLE_DIR / "analysis_results/lstm"
    checkpoint = EXAMPLE_DIR / "flip_flop_lstm.pth"
    run_analysis(checkpoint, max_iters, output_dir, model_class=LSTMModel)


if __name__ == "__main__":
    main()
