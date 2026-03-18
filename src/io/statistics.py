import json
from pathlib import Path
import numpy as np
from src.conf.config_schema import ExperimentConfig

def calculate_and_save_statistics(sim_data: dict[str, np.ndarray], output_dir: Path, config: ExperimentConfig) -> dict[str, float]:
    x = np.asarray(sim_data["states"])
    mu = np.asarray(sim_data["estimates"])
    u = np.asarray(sim_data["controls"])

    # Target equilibrium
    x_eq = np.array(config.controller.x_eq)
    
    # Calculate errors
    tracking_errors = x - x_eq
    estimation_errors = x - mu

    stats = {
        "rms_tracking_error": float(np.sqrt(np.mean(np.linalg.norm(tracking_errors, axis=1)**2))),
        "rms_estimation_error": float(np.sqrt(np.mean(np.linalg.norm(estimation_errors, axis=1)**2))),
        "rms_control_effort": float(np.sqrt(np.mean(np.linalg.norm(u, axis=1)**2))),
    }

    with (output_dir / config.directories.statistics_filename).open("w") as f:
        json.dump(stats, f, indent=4)

    return stats