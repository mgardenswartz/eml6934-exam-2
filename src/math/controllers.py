from typing import Callable
import jax
import jax.numpy as jnp
import equinox as eqx
from jaxtyping import Array, Float
import numpy as np
import scipy.linalg as sla

class OpenLoopRectanglePolicy(eqx.Module):
    dt: float

    @eqx.filter_jit
    def compute_control_input(self, step_idx: int) -> Float[Array, "3"]:
        steps_per_leg = int(5.0 / self.dt)
        index = step_idx % steps_per_leg
        
        turn1_idx = 2 * int(1.0 / self.dt)
        turn2_idx = 4 * int(1.0 / self.dt)
        
        u_turn1 = jnp.array([jnp.deg2rad(45), 100 * self.dt, jnp.deg2rad(45)])
        u_turn2 = jnp.array([jnp.deg2rad(45), 0.0, jnp.deg2rad(45)])
        u_straight = jnp.array([0.0, 100 * self.dt, 0.0])
        
        u = jnp.where(index == turn1_idx, u_turn1, 
              jnp.where(index == turn2_idx, u_turn2, u_straight))
        return u

class PolarPostureController(eqx.Module):
    k_rho: float
    k_alpha: float
    k_beta: float
    x_eq: Float[Array, "nx"]

    @eqx.filter_jit
    def compute_control_input(self, x_hat: Float[Array, "nx"]) -> Float[Array, "nu"]:
        dx = x_hat[0] - self.x_eq[0]
        dy = x_hat[1] - self.x_eq[1]
        theta = x_hat[2]

        # 1. Convert to Polar coordinates
        rho = jnp.sqrt(dx**2 + dy**2)
        theta_target = jnp.arctan2(-dy, -dx)
        
        def wrap(angle):
            return (angle + jnp.pi) % (2 * jnp.pi) - jnp.pi

        alpha = wrap(theta_target - theta)
        beta = wrap(self.x_eq[2] - theta_target)

        # 2. Astolfi Non-Linear Control Laws
        v = self.k_rho * rho * jnp.cos(alpha)
        omega = self.k_alpha * alpha + self.k_beta * beta

        # 3. Handle Origin Singularity (stop translating, just turn to final heading)
        v = jnp.where(rho < 1e-3, 0.0, v)
        omega = jnp.where(rho < 1e-3, self.k_alpha * wrap(self.x_eq[2] - theta), omega)

        # 4. Map to your specific control vector [u_1, u_2, u_3]
        # Your kinematics: u_2 is forward velocity, u_1 + u_3 is turning.
        u_1 = omega
        u_2 = v
        u_3 = 0.0

        return jnp.array([u_1, u_2, u_3], dtype=jnp.float32)

class LQR(eqx.Module):
    # (Leaving LQR intact as requested)
    K: Float[Array, "nu nx"]
    x_eq: Float[Array, "nx"]
    u_eq: Float[Array, "nu"]

    @eqx.filter_jit
    def compute_control_input(self, x_hat: Float[Array, "nx"]) -> Float[Array, "nu"]:
        delta_x = x_hat - self.x_eq
        delta_u = -self.K @ delta_x
        return self.u_eq + delta_u

def design_infinite_horizon_lqr(
    f_sys: Callable, Q: Float[Array, "nx nx"], R: Float[Array, "nu nu"],
    x_eq: Float[Array, "nx"], u_eq: Float[Array, "nu"], is_discrete: bool
) -> LQR:
    A_jax = jax.jacfwd(f_sys, argnums=0)(x_eq, u_eq)
    B_jax = jax.jacfwd(f_sys, argnums=1)(x_eq, u_eq)
    A, B = np.array(A_jax), np.array(B_jax)
    Q_np, R_np = np.array(Q), np.array(R)

    if is_discrete:
        P = sla.solve_discrete_are(A, B, Q_np, R_np)
        term1 = np.linalg.inv(R_np + B.T @ P @ B)
        K_np = term1 @ B.T @ P @ A
    else:
        P = sla.solve_continuous_are(A, B, Q_np, R_np)
        K_np = np.linalg.inv(R_np) @ B.T @ P

    return LQR(K=jnp.array(K_np, dtype=jnp.float32), x_eq=x_eq, u_eq=u_eq)