import jax
import jax.numpy as jnp
from src.conf.config_schema import ExperimentConfig
from src.math.dynamics import f, h
from src.math.controllers import PolarPostureController
from src.math.filters import EKF

def run_simulation(config: ExperimentConfig) -> dict[str, jax.Array]:
    nx, nu, nz = 3, 3, 1
    dt = config.simulation.dt
    num_steps = int(config.simulation.duration_seconds / dt)
    t_array = jnp.arange(num_steps) * dt

    x_eq = jnp.array(config.controller.x_eq, dtype=jnp.float32)

    # Initialize the new Non-Linear Controller
    controller = PolarPostureController(
        k_rho=config.controller.k_rho,
        k_alpha=config.controller.k_alpha,
        k_beta=config.controller.k_beta,
        x_eq=x_eq
    )

    Q_ekf = config.filter.q_ekf * jnp.eye(nz)
    R_ekf = config.filter.r_ekf * jnp.eye(nx)
    ekf = EKF(f_sys=f, h_sys=h, Q=Q_ekf, R=R_ekf, is_discrete=True)

    noise_cov = config.simulation.noise_std * jnp.eye(Q_ekf.shape[0]) # diagonal for simplicity
    disturbance_cov = config.simulation.disturbance_std * jnp.eye(R_ekf.shape[0]) # diagonal for simplicity

    x0 = jnp.array(config.simulation.x0, dtype=jnp.float32)
    mu0 = jnp.zeros_like(x0) 
    Sigma0 = jnp.zeros_like(R_ekf)
    key = jax.random.PRNGKey(config.simulation.random_seed)

    def step_fn(carry, t):
        x, mu, Sigma, rng = carry
        rng, k_proc, k_meas = jax.random.split(rng, 3)

        # 1. Calculate Control Law using EKF state
        u = controller.compute_control_input(mu)

        # 2. Advance True Plant
        w = jax.random.multivariate_normal(k_proc, jnp.zeros(nx), disturbance_cov)
        x_next = f(x, u) + w

        # 3. Simulate Sensor
        v = jax.random.multivariate_normal(k_meas, jnp.zeros(nz), noise_cov)
        z = h(x_next) + v

        # 4. EKF Update
        mu_next, Sigma_next = ekf.propagate(mu, Sigma, u, z, dt)

        data = {"time": t, "states": x, "estimates": mu, "controls": u}
        return (x_next, mu_next, Sigma_next, rng), data

    _, history = jax.lax.scan(step_fn, (x0, mu0, Sigma0, key), t_array)
    return history