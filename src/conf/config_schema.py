from dataclasses import dataclass, field
import jax.numpy as jnp

@dataclass
class DirectoriesConfig:
    output_parent_dir: str = "output"
    figures_dir: str = "figures"
    statistics_filename: str = "statistics.json"

@dataclass
class SimulationConfig:
    dt: float = 0.1
    num_steps: int = 200
    x0: list[float] = field(default_factory=lambda: [180.0, 50.0, 0.0])
    random_seed: int = 42

@dataclass
class NoiseConfig:
    alphas: list[float] = field(default_factory=lambda: [0.05**2, 0.005**2, 0.1**2, 0.01**2])
    beta: float = 0.007615435  # deg2rad(5)**2

@dataclass
class EnvironmentConfig:
    # 6 markers as defined in the MATLAB template
    marker_pos: list[list[float]] = field(default_factory=lambda: [
        [21.0, 0.0],
        [242.0, 0.0],
        [463.0, 0.0],
        [463.0, 292.0],
        [242.0, 292.0],
        [21.0, 292.0]
    ])

@dataclass
class PlotSettingsConfig:
    aspect_ratio_x: int = 10
    aspect_ratio_y: int = 6
    dpi: int = 300
    label_font_size: int = 12
    save_extension: str = "pdf"
    show_figures: bool = False

@dataclass
class ExperimentConfig:
    directories: DirectoriesConfig = field(default_factory=DirectoriesConfig)
    simulation: SimulationConfig = field(default_factory=SimulationConfig)
    noise: NoiseConfig = field(default_factory=NoiseConfig)
    environment: EnvironmentConfig = field(default_factory=EnvironmentConfig)
    plot_settings: PlotSettingsConfig = field(default_factory=PlotSettingsConfig)