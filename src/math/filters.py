from typing import Callable, cast
import jax
import jax.numpy as jnp
import jax.scipy as jsp
import equinox as eqx
from jaxtyping import Array, Float
import diffrax

class EKF(eqx.Module):
    f_sys: Callable
    h_sys: Callable
    Q: Float[Array, "nz nz"]
    R: Float[Array, "nx nx"]
    is_discrete: bool = eqx.field(static=True)

    @eqx.filter_jit
    def propagate(
        self,
        mu_prev: Float[Array, "nx"], 
        Sigma_prev: Float[Array, "nx nx"],
        u: Float[Array, "nu"],
        z: Float[Array, "nz"],
        dt: float
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

        Sigma_overbar_now = self.R + G @ Sigma_prev @ G.T

        dh_dx = jax.jacfwd(self.h_sys)(mu_overbar_now)
        temp_prod = Sigma_overbar_now @ dh_dx.T
        K_kalman = temp_prod @ jnp.linalg.inv(dh_dx @ temp_prod + self.Q)
        
        mu_now = mu_overbar_now + K_kalman @ (z - self.h_sys(mu_overbar_now))
        I_KH = jnp.eye(nx) - K_kalman @ dh_dx
        Sigma_now = I_KH @ Sigma_overbar_now @ I_KH.T + K_kalman @ self.Q @ K_kalman.T
        
        return mu_now, Sigma_now