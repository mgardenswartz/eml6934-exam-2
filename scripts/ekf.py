import jax
import jax.numpy as jnp
import equinox as eqx
from typing import Callable
from jaxtyping import Array, Float
import diffrax
from sympy import numer

class EKF(eqx.Module):
    f_sys: Callable  # Continuous-time dynamical model
    h_sys: Callable  # Continuous-time measurement model
    Q: Float[Array, "nz nz"]  # Measurement noise (PD? PSD? symmetric?)
    R: Float[Array, "nx nx"]  # Process noise (PD? PSD? symmetric?)
    
    def step(
        self,
        mu_prev: Float[Array, "nx"], 
        Sigma_prev: Float[Array, "nx nx"],
        u: Float[Array, "nu"], 
        z: Float[Array, "nz"],
        t_prev: float,
        dt: float
    ) -> tuple[Float[Array, "nx"], Float[Array, "nx nx"]]:
        mu_overbar, Sigma_overbar = self.predict(
            mu_prev=mu_prev,
            Sigma_prev=Sigma_prev,
            t_prev=t_prev,
            u=u,
            dt=dt
        )
        return mu_overbar, Sigma_overbar
        # x_new, Sigma_new = self.update(x_pred, Sigma_pred, z)
        # return x_new, Sigma_new


    # def update(self, x_pred, Sigma_pred, z):
    #     # H is the Jacobian of h_sys with respect to the first argument (x)
    #     # evaluated at x_pred.
    #     H = jax.jacfwd(self.h_sys)(x_pred)

    def predict(
        self,
        mu_prev: Float[Array, "nx"],
        Sigma_prev: Float[Array, "nx nx"],
        u: Float[Array, "nu"],
        t_prev: float,
        dt: float
    ) -> tuple[Float[Array, "nx"], Float[Array, "nx nx"]]:
        
        # propagate mean
        def vector_field(t, x, args): return self.f_sys(x, u)
        ode_term = diffrax.ODETerm(vector_field)
        solver = diffrax.Dopri5() # Runge-Kutta 4/5
        sol = diffrax.diffeqsolve(
            ode_term,
            solver, 
            t0=t_prev, 
            t1=(t_prev + dt), # Is this right? It's not t_prev + 10 * dt or anything right?
            dt0=dt, 
            y0=mu_prev
        )
        if sol.result != diffrax.RESULTS.successful:
            raise RuntimeError(f"SIMULATION FAILED: Diffrax Error {sol.result}")
        mu_overbar_now = sol.ys[-1] # Is the [-1] needed? Can the "None" typecast be fixed here?
        
        # propagate covariance
        df_dx = jax.jacfwd(self.f_sys)(mu_prev, u)
        Sigma_overbar_now = self.R + jnp.matmul( # Does this support jit? Is this the right way to write a three-matrix prod?
            jnp.matmul(df_dx, Sigma_prev),
            df_dx.T
        )

        return mu_overbar_now, Sigma_overbar_now

def f(x_prev: Float[Array, "nx"], u: Float[Array, "nu"] ) -> Float[Array, "nx"]:
    x_1 = x_prev[0]
    x_2 = x_prev[1]
    x_3 = x_prev[2]
    u_1 = u[0]
    u_2 = u[1]
    u_3 = u[2]
    x_now = jnp.array([
        [x_1 + u_2 * jnp.cos(x_3 + u_1)], 
        [x_2 + u_2 * jnp.sin(x_3 + u_1)],
        [x_3 + u_3 + u_1]
    ], dtype=jnp.float32
    )
    return x_now

def h(x_prev: Float[Array, "nx"]) -> Float[Array, "ny"]:
    L_x = 0
    L_y = 0
    x_1 = x_prev[0]
    x_2 = x_prev[1]
    x_3 = x_prev[2]
    numerator = L_y - x_2
    denominator = L_x - x_1
    # if denominator == 0:
    #     ans = jnp.array([jnp.pi / 2], dtype=jnp.float32)
    #     return -ans * jnp.sign(numerator) - x_3 # WTF why negative? How will jnp.sign affect diffrax, also?
    
    h_x = jnp.arctan(numerator / denominator) - x_3
    return jnp.array([h_x], dtype=jnp.float32) # rad

def main() -> None:
    u_0 = jnp.array([1, 2, 3])
    x_0 = jnp.array([0, 5, 0])
    ans = f(x_prev=x_0, u=u_0)
    print(ans)
    ans = h(x_prev=x_0)
    print(ans)


if __name__ == "__main__":
    main()
