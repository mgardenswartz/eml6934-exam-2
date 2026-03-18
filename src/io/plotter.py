import pickle
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from src.conf.config_schema import ExperimentConfig

def generate_all_plots(
    data_filepath: Path, 
    figures_dir: Path, 
    config: ExperimentConfig,
    plot_title: str,
    filename: str
) -> None:
    figures_dir.mkdir(parents=True, exist_ok=True)
    with data_filepath.open("rb") as f:
        data = pickle.load(f)

    x = np.asarray(data["states"])
    mu = np.asarray(data["estimates"])
    markers = np.array(config.environment.marker_pos)

    fig, ax = plt.subplots(figsize=(config.plot_settings.aspect_ratio_x, config.plot_settings.aspect_ratio_y))
    
    for i, marker in enumerate(markers):
        ax.scatter(marker[0], marker[1], marker='s', s=100, facecolors='none', edgecolors='k')
        ax.text(marker[0], marker[1] + 15, str(i+1), ha='center')

    ax.plot(x[:, 0], x[:, 1], label="True Robot Path", color='blue', linewidth=1.5)
    
    # Check if it's PF or EKF based on title
    filter_label = "PF Filter Path" if "Particle" in plot_title else "EKF Filter Path"
    ax.plot(mu[:, 0], mu[:, 1], label=filter_label, color='red', linestyle='--', linewidth=1.5)
    
    ax.set_xlabel("X Position")
    ax.set_ylabel("Y Position")
    ax.set_title(plot_title)
    ax.grid(True, linestyle=':')
    ax.legend()
    ax.axis('equal') 
    
    fig.savefig(figures_dir / f"{filename}.{config.plot_settings.save_extension}", dpi=config.plot_settings.dpi)
    plt.close(fig)