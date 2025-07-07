import jax
import jax.numpy as jnp
from typing import Callable, List


def nguyen_9(x: jnp.ndarray) -> jnp.ndarray:
    return jnp.sin(x[:, 0]) + jnp.sin(x[:, 1] ** 2)


def nguyen_10(x: jnp.ndarray) -> jnp.ndarray:
    return 2 * jnp.sin(x[:, 0]) * jnp.cos(x[:, 1])


def nguyen_12(x: jnp.ndarray) -> jnp.ndarray:
    return jnp.sin(x[:, 0]) ** 4 - jnp.sin(x[:, 0]) ** 3 + 0.5 * jnp.sin(x[:, 1]) ** 2 - jnp.sin(x[:, 1])


def combine_regression_functions(x: jnp.ndarray,
                                 regression_functions: List[Callable[[jnp.ndarray], jnp.ndarray]],
                                 file_path: str):
    y = jnp.hstack([f(x).reshape(-1, 1) for f in regression_functions])
    jnp.save(f"{file_path}_x.npy", x)
    jnp.save(f"{file_path}_y.npy", y)


def nguyen_9_12():
    file_path = "nguyen_9_12"
    key = jax.random.PRNGKey(0)
    x = jax.random.uniform(key, shape=(20, 2), minval=0.0, maxval=1.0)
    combine_regression_functions(x, [nguyen_9, nguyen_12], file_path)


def two_bits_input() -> jnp.ndarray:
    inputs_list = []
    for a in [0, 1]:
        for b in [0, 1]:
            for c in [0, 1]:
                for d in [0, 1]:
                    inputs_list.append([a, b, c, d])
    return jnp.asarray(inputs_list)


# 2-bit adder
def two_bits_adder():
    file_path = "two_bit_adder"
    jnp.save(f"{file_path}_x.npy", two_bits_input())

    outputs_list = [
        [0, 0, 0],
        [0, 0, 1],
        [0, 1, 0],
        [0, 1, 1],
        [0, 0, 1],
        [0, 1, 0],
        [0, 1, 1],
        [1, 0, 0],
        [0, 1, 0],
        [0, 1, 1],
        [1, 0, 0],
        [1, 0, 1],
        [0, 1, 1],
        [1, 0, 0],
        [1, 0, 1],
        [1, 1, 0],
    ]
    outputs = jnp.asarray(outputs_list)
    jnp.save(f"{file_path}_y.npy", outputs)


def two_bit_multiplier():
    file_path = "two_bit_multiplier"

    jnp.save(f"{file_path}_x.npy", two_bits_input())

    outputs_list = [
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 1],
        [0, 0, 1, 0],
        [0, 0, 1, 1],
        [0, 0, 0, 0],
        [0, 0, 1, 0],
        [0, 1, 0, 0],
        [0, 1, 1, 0],
        [0, 0, 0, 0],
        [0, 0, 1, 1],
        [0, 1, 1, 0],
        [1, 0, 0, 1],
    ]
    outputs = jnp.asarray(outputs_list)
    jnp.save(f"{file_path}_y.npy", outputs)


if __name__ == '__main__':
    # two_bit_multiplier()
    nguyen_9_12()
