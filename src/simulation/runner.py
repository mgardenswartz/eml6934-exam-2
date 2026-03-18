import jax
import jax.numpy as jnp
from src.conf.config_schema import ExperimentConfig
from src.math.dynamics import f, h, wrap_angle
from scripts.controllers import OpenLoopRectanglePolicy
from src.math.filters import EKF
from src.math.filters import ParticleFilter

def run_simulation(config: ExperimentConfig) -> dict[str, jax.Array]:
    num_steps = config.simulation.num_steps
    dt = config.simulation.dt
    step_array = jnp.arange(num_steps)
    
    alphas = jnp.array(config.noise.alphas)
    beta = config.noise.beta
    markers = jnp.array(config.environment.marker_pos)

    policy = OpenLoopRectanglePolicy(dt=dt)
    ekf = EKF(f_sys=f, h_sys=h, is_discrete=True)

    x0 = jnp.array(config.simulation.x0, dtype=jnp.float32)
    Sigma0 = jnp.diag(jnp.array([10.0, 10.0, 1.0]))
    key = jax.random.PRNGKey(config.simulation.random_seed)

    def step_fn(carry, step_idx):
        x_real, mu, Sigma, rng = carry
        rng, k_proc, k_meas = jax.random.split(rng, 3)

        # --- THE POLICY ---
        # 1. Get noise-free control
        u_nf = policy.compute_control_input(step_idx)

        # Calculate Control Noise Covariance (M) based on current action
        rot1, trans, rot2 = u_nf[0], u_nf[1], u_nf[2]
        M = jnp.diag(jnp.array([
            alphas[0] * rot1**2 + alphas[1] * trans**2,
            alphas[2] * trans**2 + alphas[3] * (rot1**2 + rot2**2),
            alphas[0] * rot2**2 + alphas[1] * trans**2
        ]))

        # --- THE TRUE PHYSICAL WORLD ---
        # 2. Add real physical noise to the wheels and advance the true robot
        std_devs = jnp.sqrt(jnp.diag(M))
        u_noisy = u_nf + std_devs * jax.random.normal(k_proc, (3,))
        x_next = f(x_real, u_noisy)

        # 3. Read the real noisy sensor bearing to the current landmark
        marker_idx = (step_idx // 2) % markers.shape[0]
        marker_pos = markers[marker_idx]
        
        z_nf = h(x_next, marker_pos)
        v = jax.random.normal(k_meas, (1,)) * jnp.sqrt(beta)
        z_real = jnp.array([wrap_angle(z_nf[0] + v[0])])

        # --- THE ESTIMATOR (EKF) ---
        # 4. Map the control noise (M) into state-space process noise (R_dynamic) using Jacobian V
        V = jax.jacfwd(f, argnums=1)(mu, u_nf)
        R_dynamic = V @ M @ V.T

        # 5. Define how the EKF should calculate innovation for an angle
        def angle_residual(z_act, z_exp):
            return jnp.array([wrap_angle(z_act[0] - z_exp[0])])

        # 6. Call the pristine EKF template
        mu_next, Sigma_next = ekf.propagate(
            mu_prev=mu, 
            Sigma_prev=Sigma, 
            u=u_nf, 
            z=z_real, 
            Q=jnp.array([[beta]]), 
            R=R_dynamic,
            dt=dt,
            h_args=(marker_pos,), 
            residual_fn=angle_residual
        )

        # 7. Ensure the final estimated heading stays within [-pi, pi]
        mu_next = mu_next.at[2].set(wrap_angle(mu_next[2]))

        data = {
            "time": step_idx * dt, 
            "states": x_real, 
            "estimates": mu_next
        }
        return (x_next, mu_next, Sigma_next, rng), data

    # Run the fast compiled loop
    _, history = jax.lax.scan(step_fn, (x0, x0, Sigma0, key), step_array)
    
    return history


def run_pf_simulation(config: ExperimentConfig) -> dict[str, jax.Array]:
    num_steps = config.simulation.num_steps
    dt = config.simulation.dt
    step_array = jnp.arange(num_steps)
    
    alphas = jnp.array(config.noise.alphas)
    beta = config.noise.beta
    markers = jnp.array(config.environment.marker_pos)

    policy = OpenLoopRectanglePolicy(dt=dt)
    pf = ParticleFilter(f_sys=f, h_sys=h, num_particles=config.simulation.num_particles)

    x0 = jnp.array(config.simulation.x0, dtype=jnp.float32)
    key = jax.random.PRNGKey(config.simulation.random_seed)
    
    # Initialize particle swarm around x0
    key, k_init = jax.random.split(key)
    Sigma0 = jnp.diag(jnp.array([10.0, 10.0, 1.0]))
    particles0 = jax.random.multivariate_normal(k_init, x0, Sigma0, shape=(config.simulation.num_particles,))

    def pf_step_fn(carry, step_idx):
        x_real, mu, particles, rng = carry
        rng, k_proc, k_meas, k_pf = jax.random.split(rng, 4)

        # 1. Get noise-free control
        u_nf = policy.compute_control_input(step_idx)

        # 2. Control Noise Covariance (M)
        rot1, trans, rot2 = u_nf[0], u_nf[1], u_nf[2]
        M = jnp.diag(jnp.array([
            alphas[0] * rot1**2 + alphas[1] * trans**2,
            alphas[2] * trans**2 + alphas[3] * (rot1**2 + rot2**2),
            alphas[0] * rot2**2 + alphas[1] * trans**2
        ]))

        # 3. True Physical World
        std_devs = jnp.sqrt(jnp.diag(M))
        u_noisy = u_nf + std_devs * jax.random.normal(k_proc, (3,))
        x_next = f(x_real, u_noisy)

        marker_idx = (step_idx // 2) % markers.shape[0]
        marker_pos = markers[marker_idx]
        
        z_nf = h(x_next, marker_pos)
        v = jax.random.normal(k_meas, (1,)) * jnp.sqrt(beta)
        z_real = jnp.array([wrap_angle(z_nf[0] + v[0])])

        def angle_residual(z_act, z_exp):
            return jnp.array([wrap_angle(z_act[0] - z_exp[0])])

        # 4. Particle Filter Update
        mu_next, particles_next = pf.propagate(
            particles=particles,
            u=u_nf,
            z=z_real,
            M=M,
            Q_val=beta,
            key=k_pf,
            h_args=(marker_pos,),
            residual_fn=angle_residual
        )

        data = {
            "time": step_idx * dt, 
            "states": x_real, 
            "estimates": mu_next
        }
        return (x_next, mu_next, particles_next, rng), data

    _, history = jax.lax.scan(pf_step_fn, (x0, x0, particles0, key), step_array)
    return history