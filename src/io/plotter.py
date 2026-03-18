import pickle
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from src.conf.config_schema import ExperimentConfig

def generate_all_plots(data_filepath: Path, figures_dir: Path, config: ExperimentConfig) -> None:
    figures_dir.mkdir(parents=True, exist_ok=True)
    with data_filepath.open("rb") as f:
        data = pickle.load(f)

    t = np.asarray(data["time"])
    x = np.asarray(data["states"])
    mu = np.asarray(data["estimates"])
    u = np.asarray(data["controls"])

    # States vs Estimates
    fig, ax = plt.subplots(figsize=(config.plot_settings.aspect_ratio_x, config.plot_settings.aspect_ratio_y))
    for i in range(x.shape[1]):
        ax.plot(t, x[:, i], label=r"True State $x_{i+1}$", alpha=0.7)
        ax.plot(t, mu[:, i], label=r"Estimate $\mu_{i+1}$", linestyle="--")
    ax.set_xlabel("Time (s)")
    ax.legend()
    ax.grid(True, linestyle='--')
    fig.savefig(figures_dir / f"states_vs_estimates.{config.plot_settings.save_extension}", dpi=config.plot_settings.dpi)
    plt.close(fig)

    # Control Inputs
    fig, ax = plt.subplots(figsize=(config.plot_settings.aspect_ratio_x, config.plot_settings.aspect_ratio_y))
    for i in range(u.shape[1]):
        ax.plot(t, u[:, i], label=r"Control $u_{i+1}$")
    ax.set_xlabel("Time (s)")
    ax.legend()
    ax.grid(True, linestyle='--')
    fig.savefig(figures_dir / f"controls.{config.plot_settings.save_extension}", dpi=config.plot_settings.dpi)
    plt.close(fig)