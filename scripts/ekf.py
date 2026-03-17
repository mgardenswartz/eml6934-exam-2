from typing import Callable, cast

import jax
import jax.numpy as jnp
import equinox as eqx
from jaxtyping import Array, Float
import diffrax

class EKF(eqx.Module):
    f_sys: Callable  # Dynamical model
    h_sys: Callable  # Measurement model
    Q: Float[Array, "nz nz"]  # Measurement noise (PD symmetric)
    R: Float[Array, "nx nx"]  # Process noise (PD symmetric)

    # evaluate at compile-time, not runtime
    already_discrete: bool = eqx.field(static=True)

    @eqx.filter_jit
    def propagate(
        self,
        mu_prev: Float[Array, "nx"], 
        Sigma_prev: Float[Array, "nx nx"],
        u: Float[Array, "nu"], # control input
        z: Float[Array, "nz"], # measurement
        dt: float
    ) -> tuple[Float[Array, "nx"], Float[Array, "nx nx"]]:

        # predict: propagate mean
        if not self.already_discrete:
            def vector_field(t: float, x: Array, args) -> Array:
                return self.f_sys(x, u)

            ode_term = diffrax.ODETerm(vector_field) # type: ignore
            solver = diffrax.Dopri5() # Runge-Kutta 4/5
            sol = diffrax.diffeqsolve(
                ode_term,
                solver, 
                t0=0.0, 
                t1=dt,
                dt0=dt, 
                y0=mu_prev
            )
            mu_overbar_now = cast(Array, sol.ys)[-1]
        else:
            mu_overbar_now = self.f_sys(mu_prev, u)

        # predict: propagate covariance
        df_dx = jax.jacfwd(self.f_sys)(mu_prev, u)
        nx = mu_prev.shape[0]
        if not self.already_discrete:
            G = jnp.eye(nx) + df_dx * dt  # assume zero-order hold
        else:
            G = df_dx

        Sigma_overbar_now = self.R + G @ Sigma_prev @ G.T

        # measurement update
        dh_dx = jax.jacfwd(self.h_sys)(mu_overbar_now) # note: not mu_prev
        temp_prod = Sigma_overbar_now @ dh_dx.T
        K_kalman = temp_prod @ jnp.linalg.inv(dh_dx @ temp_prod + self.Q)
        mu_now = mu_overbar_now + K_kalman @ (z - self.h_sys(mu_overbar_now))
        I_KH = jnp.eye(nx) - K_kalman @ dh_dx  # Joseph form for numeric stability, suggested by Gemini 
        Sigma_now = I_KH @ Sigma_overbar_now @ I_KH.T + K_kalman @ self.Q @ K_kalman.T
        return mu_now, Sigma_now


def f(x_prev: Float[Array, "nx"], u: Float[Array, "nu"] ) -> Float[Array, "nx"]:
    x_1 = x_prev[0]
    x_2 = x_prev[1]
    x_3 = x_prev[2]
    u_1 = u[0]
    u_2 = u[1]
    u_3 = u[2]
    x_now = jnp.array([
        x_1 + u_2 * jnp.cos(x_3 + u_1), 
        x_2 + u_2 * jnp.sin(x_3 + u_1),
        x_3 + u_3 + u_1
    ], dtype=jnp.float32
    )
    return x_now


def h(x_prev: Float[Array, "nx"]) -> Float[Array, "nz"]:
    L_x = 0
    L_y = 0
    x_1 = x_prev[0]
    x_2 = x_prev[1]
    x_3 = x_prev[2]
    h_x = jnp.arctan2(L_y - x_2, L_x - x_1) - x_3
    return jnp.array([h_x], dtype=jnp.float32) # rad


def main() -> None:
    q = 0.01
    r = 0.02

    u_0 = jnp.array([1, 2, 3], dtype=jnp.float32)
    x_0 = jnp.array([0, 5, 0], dtype=jnp.float32)
    z = jnp.array([0], dtype=jnp.float32)
    dt = 0.5

    nx = x_0.shape[0]
    nz = z.shape[0]
    Q = q * jnp.eye(nz, dtype=jnp.float32)
    R = r * jnp.eye(nx, dtype=jnp.float32)
    
    # assume perfect information initially
    Sigma_0 = jnp.zeros_like(R, dtype=jnp.float32)
    mu_0 = x_0

    # call filter
    my_ekf = EKF(f_sys=f, h_sys=h, Q=Q, R=R, already_discrete=True)
    mu, Sigma = my_ekf.propagate(
        mu_prev=mu_0,
        Sigma_prev=Sigma_0,
        u=u_0,
        z=z,
        dt=dt
    )
    print(f"New mean and covariance are '{mu}', \n'{Sigma}'")

if __name__ == "__main__":
    main()
