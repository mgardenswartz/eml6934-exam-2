from typing import Callable, cast, Optional
import jax
import jax.numpy as jnp
import jax.scipy as jsp
import equinox as eqx
from jaxtyping import Array, Float
import diffrax

class EKF(eqx.Module):
    f_sys: Callable
    h_sys: Callable
    is_discrete: bool = eqx.field(static=True)

    @eqx.filter_jit
    def propagate(
        self,
        mu_prev: Float[Array, "nx"], 
        Sigma_prev: Float[Array, "nx nx"],
        u: Float[Array, "nu"],
        z: Float[Array, "nz"],
        Q: Float[Array, "nz nz"],
        R: Float[Array, "nx nx"],
        dt: float,
        h_args: tuple = (),
        residual_fn: Optional[Callable] = None
    ) -> tuple[Float[Array, "nx"], Float[Array, "nx nx"]]:

        if not self.is_discrete:
            def vector_field(t: float, x: Array, args) -> Array:
                return self.f_sys(x, u)
            ode_term = diffrax.ODETerm(vector_field)
            solver = diffrax.Dopri5()
            sol = diffrax.diffeqsolve(ode_term, solver, t0=0.0, t1=dt, dt0=dt, y0=mu_prev)
            mu_overbar_now = cast(Array, sol.ys)[-1]
        else:
            mu_overbar_now = self.f_sys(mu_prev, u)

        df_dx = jax.jacfwd(self.f_sys, argnums=0)(mu_prev, u)
        nx = mu_prev.shape[0]
        
        if not self.is_discrete:
            G = jsp.linalg.expm(df_dx * dt)
        else:
            G = df_dx

        Sigma_overbar_now = R + G @ Sigma_prev @ G.T

        dh_dx = jax.jacfwd(self.h_sys)(mu_overbar_now, *h_args) 
        temp_prod = Sigma_overbar_now @ dh_dx.T
        K_kalman = temp_prod @ jnp.linalg.inv(dh_dx @ temp_prod + Q)
        
        z_expected = self.h_sys(mu_overbar_now, *h_args)
        if residual_fn is not None:
            innovation = residual_fn(z, z_expected)
        else:
            innovation = z - z_expected
            
        mu_now = mu_overbar_now + K_kalman @ innovation
        I_KH = jnp.eye(nx) - K_kalman @ dh_dx
        Sigma_now = I_KH @ Sigma_overbar_now @ I_KH.T + K_kalman @ Q @ K_kalman.T
        
        return mu_now, Sigma_now
    

class ParticleFilter(eqx.Module):
    f_sys: Callable
    h_sys: Callable
    num_particles: int = eqx.field(static=True)

    @eqx.filter_jit
    def propagate(
        self,
        particles: Float[Array, "N nx"],
        u: Float[Array, "nu"],
        z: Float[Array, "nz"],
        M: Float[Array, "nu nu"], # Control noise covariance
        Q_val: float,             # Scalar measurement noise variance
        key: jax.Array,
        h_args: tuple = (),
        residual_fn: Optional[Callable] = None
    ) -> tuple[Float[Array, "nx"], Float[Array, "N nx"]]:
        
        key_motion, key_resample = jax.random.split(key)

        # predict
        u_noise = jax.random.multivariate_normal(
            key_motion, jnp.zeros_like(u), M, shape=(self.num_particles,)
        )
        u_particles = u + u_noise
        vmap_f = jax.vmap(self.f_sys, in_axes=(0, 0))
        particles_bar = vmap_f(particles, u_particles)

        # update
        vmap_h = jax.vmap(self.h_sys, in_axes=(0, *[None]*len(h_args)))
        z_expected = vmap_h(particles_bar, *h_args)

        if residual_fn is not None:
            vmap_res = jax.vmap(residual_fn, in_axes=(None, 0))
            innov = vmap_res(z, z_expected)
        else:
            innov = z - z_expected

        innov = innov.reshape(-1)
        weights = jnp.exp(-0.5 * (innov**2) / Q_val)
        weights = weights + 1e-8 # Prevent division by zero
        weights = weights / jnp.sum(weights)

        r = jax.random.uniform(key_resample, minval=0.0, maxval=1.0 / self.num_particles)
        c = jnp.cumsum(weights)
        U = r + jnp.arange(self.num_particles) * (1.0 / self.num_particles)

        indices = jnp.searchsorted(c, U) 
        particles_resampled = particles_bar[indices]
        mean_pos = jnp.mean(particles_resampled[:, :2], axis=0)
        
        # Circular mean for the heading angle
        mean_theta = jnp.arctan2(
            jnp.mean(jnp.sin(particles_resampled[:, 2])), 
            jnp.mean(jnp.cos(particles_resampled[:, 2]))
        )
        mu_now = jnp.array([mean_pos[0], mean_pos[1], mean_theta])

        return mu_now, particles_resampled