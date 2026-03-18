import os
from pathlib import Path
from src.conf.config_schema import ExperimentConfig
from src.io.data_exporter import export_to_pickle
from src.io.plotter import generate_all_plots
from src.simulation.runner import run_simulation, run_pf_simulation

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2" 

def main() -> None:
    config = ExperimentConfig()
    output_dir = Path(config.directories.output_parent_dir) / "assignment_2"
    figures_dir = output_dir / config.directories.figures_dir

    print("Running EKF Localization...")
    ekf_data = run_simulation(config)
    export_to_pickle(ekf_data, output_dir, "ekf_data.pkl")
    generate_all_plots(
        output_dir / "ekf_data.pkl", figures_dir, config, 
        "Assignment 2.1(b): EKF Localization", "ekf_path"
    )
    
    print("Running Particle Filter Localization...")
    pf_data = run_pf_simulation(config)
    export_to_pickle(pf_data, output_dir, "pf_data.pkl")
    generate_all_plots(
        output_dir / "pf_data.pkl", figures_dir, config, 
        "Assignment 2.2(b): Particle Filter Localization", "pf_path"
    )
    
    print(f"\nSuccess! Plots saved to: {figures_dir}")

if __name__ == "__main__":
    main()