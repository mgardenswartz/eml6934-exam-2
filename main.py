import os
import platform
from pathlib import Path
import jax
import optuna
import numpy as np

from src.conf.config_schema import ExperimentConfig
from src.io.data_exporter import export_to_pickle
from src.io.plotter import generate_all_plots
from src.io.statistics import calculate_and_save_statistics
from src.simulation.runner import run_simulation

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2" 

def objective(trial: optuna.Trial) -> float:
    # 1. Optuna suggests polar gains instead of LQR weights
    k_rho = trial.suggest_float("k_rho", 0.1, 5.0)
    k_alpha = trial.suggest_float("k_alpha", 0.1, 10.0)
    k_beta = trial.suggest_float("k_beta", -5.0, 5.0) 
    q_ekf = trial.suggest_float("q_ekf", 0.001, 1.0, log=True)
    r_ekf = trial.suggest_float("r_ekf", 0.001, 1.0, log=True)

    config = ExperimentConfig()
    config.controller.k_rho = k_rho
    config.controller.k_alpha = k_alpha
    config.controller.k_beta = k_beta
    config.filter.q_ekf = q_ekf
    config.filter.r_ekf = r_ekf

    output_dir = Path(config.directories.output_parent_dir) / f"trial_{trial.number}"
    
    try:
        sim_data = run_simulation(config)
    except (RuntimeError, np.linalg.LinAlgError):
        return 1e9 

    output_dir.mkdir(parents=True, exist_ok=True)
    stats = calculate_and_save_statistics(sim_data, output_dir, config)

    cost = stats["rms_tracking_error"] + stats["rms_estimation_error"] + 0.01 * stats["rms_control_effort"]
    return cost

def run_best_parameters(best_params: dict[str, float]) -> None:
    print(f"\n{'='*40}\nGENERATING BEST RUN PLOTS\n{'-'*40}")
    
    config = ExperimentConfig()
    config.controller.k_rho = best_params["k_rho"]
    config.controller.k_alpha = best_params["k_alpha"]
    config.controller.k_beta = best_params["k_beta"]
    config.filter.q_ekf = best_params["q_ekf"]
    config.filter.r_ekf = best_params["r_ekf"]

    output_dir = Path(config.directories.output_parent_dir) / "best_run"
    figures_dir = output_dir / config.directories.figures_dir

    sim_data = run_simulation(config)
    
    export_to_pickle(sim_data, output_dir, "simulation_data.pkl")
    generate_all_plots(output_dir / "simulation_data.pkl", figures_dir, config)
    stats = calculate_and_save_statistics(sim_data, output_dir, config)

    for key, value in stats.items():
        print(f"{key.replace('_', ' ').title()}: {value:.3f}")
    
    print(f"{'='*40}")
    print(f"Data and figures saved to: {output_dir}\n")

def main() -> None:
    backend = jax.default_backend().upper()
    print(f"\n{'='*40}")
    print(f"System: {platform.system()} detected.")
    print(f"Hardware: JAX is hardware-accelerated on {backend}.")
    print(f"{'='*40}\n")

    study = optuna.create_study(
        study_name="lqg_tuning_sweep", 
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=42)
    )
    
    print("Starting Optuna sweep...")
    study.optimize(objective, n_trials=25)

    print(f"\nSweep Complete! Best Trial: {study.best_trial.number}")
    print(f"Best Cost: {study.best_value:.3f}")
    print("Best Parameters:")
    for k, v in study.best_params.items():
        print(f"  {k}: {v:.4f}")

    run_best_parameters(study.best_params)

if __name__ == "__main__":
    main()