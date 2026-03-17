import jax
import jax.numpy as jnp
import equinox as eqx
from typing import Callable
from jaxtyping import Array, Float
import diffrax

class EKF(eqx.Module):
    f_sys: Callable  # Continuous-time dynamical model
    h_sys: Callable  # Continuous-time measurement model
    Q: Float[Array, "nz nz"]  # Measurement noise (PD symmetric)
    R: Float[Array, "nx nx"]  # Process noise (PD symmetric)
    
    def propagate(
        self,
        mu_prev: Float[Array, "nx"], 
        Sigma_prev: Float[Array, "nx nx"],
        u: Float[Array, "nu"], # control input
        z: Float[Array, "nz"], # measurement
        dt: float
    ) -> tuple[Float[Array, "nx"], Float[Array, "nx nx"]]:
        # predict: propagate mean
        def vector_field(t, x, args): return self.f_sys(x, u)
        ode_term = diffrax.ODETerm(vector_field)
        solver = diffrax.Dopri5() # Runge-Kutta 4/5
        sol = diffrax.diffeqsolve(
            ode_term,
            solver, 
            t0=0.0, 
            t1=dt,
            dt0=dt, 
            y0=mu_prev
        )
        if sol.result != diffrax.RESULTS.successful:
            raise RuntimeError(f"SIMULATION FAILED: Diffrax Error {sol.result}")
        mu_overbar_now = sol.ys[-1] # Can the "None" typecast be fixed here? It's still appearing.s
        
        # predict: propagate covariance
        df_dx = jax.jacfwd(self.f_sys)(mu_prev, u)
        Sigma_overbar_now = self.R + df_dx @ Sigma_prev @ df_dx.T
    
        # measurement update
        dh_dx = jax.jacfwd(self.h_sys)(mu_overbar_now) # Note: not mu_prev
        temp_prod = Sigma_overbar_now @ dh_dx.T
        K_kalman = temp_prod @ jnp.linalg.inv(dh_dx @ temp_prod + self.Q)
        mu_now = mu_overbar_now + K_kalman @ (z - self.h_sys(mu_overbar_now))
        nx = Sigma_overbar_now.shape[0]
        Sigma_now = (jnp.eye(nx) - K_kalman @ dh_dx) @ Sigma_overbar_now
        return mu_now, Sigma_now

def f(x_prev: Float[Array, "nx"], u: Float[Array, "nu"] ) -> Float[Array, "nx"]:
    x_1 = x_prev[0]
    x_2 = x_prev[1]
    x_3 = x_prev[2]
    u_1 = u[0]
    u_2 = u[1]
    u_3 = u[2]
    x_dot = jnp.array([
        x_1 + u_2 * jnp.cos(x_3 + u_1), 
        x_2 + u_2 * jnp.sin(x_3 + u_1),
        x_3 + u_3 + u_1
    ], dtype=jnp.float32
    )
    return x_dot.T

def h(x_prev: Float[Array, "nx"]) -> Float[Array, "nz"]:
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
    my_ekf = EKF(f_sys=f, h_sys=h, Q=Q, R=R)
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
