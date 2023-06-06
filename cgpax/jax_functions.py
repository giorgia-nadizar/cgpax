
from __future__ import annotations

from jax import jit
from jax.tree_util import register_pytree_node_class
from functools import partial

import jax.numpy as jnp
from jax.lax import cond, switch


@jit
def plus(x, y):
    return jnp.add(x, y)


@jit
def minus(x, y):
    return jnp.add(x, -y)


@jit
def times(x, y):
    return jnp.multiply(x, y)


@jit
def prot_div(x, y):
    def prot(x, y):
        return 1.0

    return cond(y == 0, jit(prot), jnp.divide, x, y)


@jit
def abs_val(x, y):
    return jnp.abs(x)


@jit
def exp(x, y):
    return jnp.exp(x)


@jit
def sin(x, y):
    return jnp.sin(x)


@jit
def lower(x, y):
    flag = x < y
    return 0.0 + flag


@jit
def greater(x, y):
    return 1.0 - lower(x, y)


@jit
def function_switch(idx, *operands):
    return switch(idx, list(JaxFunction.existing_functions.values()), *operands)


@register_pytree_node_class
class JaxFunction:
    existing_functions = {
        "plus": plus,
        "minus": minus,
        "times": times,
        "prot_div": prot_div,
        "abs": abs_val,
        "exp": exp,
        "sin": sin,
        "lower": lower,
        "greater": greater
    }

    arities = {
        "plus": 2,
        "minus": 2,
        "times": 2,
        "prot_div": 2,
        "abs": 1,
        "exp": 1,
        "sin": 1,
        "lower": 2,
        "greater": 2
    }

    def __init__(self, op) -> None:
        self.operator = op
        pass

    @classmethod
    def from_name(cls, name: str) -> JaxFunction:
        return cls(cls.existing_functions[name])

    @partial(jit, static_argnums=0)
    def apply(self, x, y):
        return self.operator(x, y)

    def __call__(self, *args, **kwargs):
        return self.apply(*args)

    def tree_flatten(self):
        children = ()
        aux_data = {"operator": self.operator}
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(aux_data["operator"])
