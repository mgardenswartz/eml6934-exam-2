import jax.numpy as jnp
from jaxtyping import Array, Float

def wrap_angle(angle: float | Array) -> float | Array:
    """Wraps an angle to the [-pi, pi] range."""
    return jnp.mod(angle + jnp.pi, 2 * jnp.pi) - jnp.pi

def f(x_prev: Float[Array, "3"], u: Float[Array, "3"]) -> Float[Array, "3"]:
    rot1, trans, rot2 = u[0], u[1], u[2]
    theta_new = x_prev[2] + rot1
    
    x_next = jnp.array([
        x_prev[0] + trans * jnp.cos(theta_new),
        x_prev[1] + trans * jnp.sin(theta_new),
        theta_new + rot2
    ])
    return x_next.at[2].set(wrap_angle(x_next[2]))

def h(x_prev: Float[Array, "3"], marker: Float[Array, "2"]) -> Float[Array, "1"]:
    dx = marker[0] - x_prev[0]
    dy = marker[1] - x_prev[1]
    z = jnp.arctan2(dy, dx) - x_prev[2]
    return jnp.array([wrap_angle(z)])