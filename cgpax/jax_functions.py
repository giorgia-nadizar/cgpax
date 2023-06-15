from __future__ import annotations

from jax import jit
from jax.tree_util import register_pytree_node_class
from functools import partial

import jax.numpy as jnp
from jax.lax import cond, switch


@register_pytree_node_class
class JaxFunction:

    def __init__(self, op, arity: int, symbol: str) -> None:
        self.operator = op
        self.arity = arity
        self.symbol = symbol
        pass

    @partial(jit, static_argnums=0)
    def apply(self, x, y):
        return self.operator(x, y)

    def __call__(self, x, y):
        return self.apply(x, y)

    def tree_flatten(self):
        children = ()
        aux_data = {"operator": self.operator, "arity": self.arity, "symbol": self.symbol}
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(aux_data["operator"], aux_data["arity"], aux_data["symbol"])


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


available_functions = {
    "plus": JaxFunction(plus, 2, "+"),
    "minus": JaxFunction(minus, 2, "-"),
    "times": JaxFunction(times, 2, "*"),
    "prot_div": JaxFunction(prot_div, 2, "/"),
    "abs": JaxFunction(abs_val, 1, "|.|"),
    "exp": JaxFunction(exp, 1, "exp"),
    "sin": JaxFunction(sin, 1, "sin"),
    "lower": JaxFunction(lower, 2, "<"),
    "greater": JaxFunction(greater, 2, ">")
}


@jit
def function_switch(idx, *operands):
    return switch(idx, list(available_functions.values()), *operands)
