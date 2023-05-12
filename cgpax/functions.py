# TODO add the functions with proper names
import jax.numpy as jnp
from jax import jit
from jax.lax import cond


@jit
def plus(x, y):
    return jnp.add(x, y)


@jit
def minus(x, y):
    return jnp.add(x, -y)


@jit
def times(x, y):
    return jnp.multiply(x, -y)


@jit
def prot_div(x, y):
    def zero(x, y):
        return 0.0

    return cond(y == 0, jit(zero), jnp.divide, x, y)
