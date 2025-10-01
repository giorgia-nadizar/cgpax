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


def feynman_13(q: jnp.ndarray, ef: jnp.ndarray, b: jnp.ndarray, v: jnp.ndarray, theta: jnp.ndarray) -> jnp.ndarray:
    return q * (ef + b * v * jnp.sin(theta))


def feynman_42(n: jnp.ndarray, kb: jnp.ndarray, t: jnp.ndarray, v: jnp.ndarray) -> jnp.ndarray:
    return n * kb * t / v


def feynman_43(n0: jnp.ndarray, m: jnp.ndarray, g: jnp.ndarray, x: jnp.ndarray, kb: jnp.ndarray,
               t: jnp.ndarray) -> jnp.ndarray:
    return n0 * jnp.exp(-m * g * x / (kb * t))


def feynman_48(n: jnp.ndarray, kb: jnp.ndarray, t: jnp.ndarray, v1: jnp.ndarray, v2: jnp.ndarray) -> jnp.ndarray:
    return n * kb * t * jnp.log(v2 / v1)


def feynman_61(q: jnp.ndarray, ef: jnp.ndarray, m: jnp.ndarray, omega_0: jnp.ndarray,
               omega: jnp.ndarray) -> jnp.ndarray:
    return q * ef / (m * (jnp.power(omega_0, 2) - jnp.power(omega, 2)))


def feynman_62(n0: jnp.ndarray, pd: jnp.ndarray, ef: jnp.ndarray, theta: jnp.ndarray, kb: jnp.ndarray,
               t: jnp.ndarray) -> jnp.ndarray:
    return n0 * (1 + pd * ef * jnp.cos(theta) / (kb * t))


def feynman_70(pd: jnp.ndarray, ef: jnp.ndarray, theta: jnp.ndarray) -> jnp.ndarray:
    return -pd * ef * jnp.cos(theta)


def feynman_77(g: jnp.ndarray, q: jnp.ndarray, b: jnp.ndarray, m: jnp.ndarray) -> jnp.ndarray:
    return g * q * b / (2 * m)


def feynman_100(rho: jnp.ndarray, q: jnp.ndarray, a: jnp.ndarray, m: jnp.ndarray) -> jnp.ndarray:
    return -rho * q * a / m


def feynman_13_61():
    file_path = "feynman_13_61"
    min_vals = jnp.array([1., 1., 1., 1., 1., 1., 3., 1.])
    max_vals = jnp.array([3., 3., 5., 5., 5., 3., 5., 2.])
    key = jax.random.PRNGKey(0)
    X = jax.random.uniform(key, shape=(500, 8)) * (max_vals - min_vals) + min_vals
    q = X[:, 0]
    ef = X[:, 1]
    b = X[:, 2]
    v = X[:, 3]
    theta = X[:, 4]
    m = X[:, 5]
    omega_0 = X[:, 6]
    omega = X[:, 7]
    y0 = feynman_13(q, ef, b, v, theta)
    y1 = feynman_61(q, ef, m, omega_0, omega)
    y = jnp.hstack([k.reshape(-1, 1) for k in [y0, y1]])
    jnp.save(f"{file_path}_x.npy", X)
    jnp.save(f"{file_path}_y.npy", y)


def feynman_42_43_48():
    file_path = "feynman_42_43_48"
    key = jax.random.PRNGKey(0)
    X = jax.random.uniform(key, shape=(500, 10), minval=1., maxval=5.)
    n = X[:, 0]
    t = X[:, 1]
    v = X[:, 2]
    kb = X[:, 3]
    n0 = X[:, 4]
    m = X[:, 5]
    x = X[:, 6]
    g = X[:, 7]
    v1 = X[:, 8]
    v2 = X[:, 9]

    y0 = feynman_42(n, kb, t, v)
    y1 = feynman_43(n0, m, g, x, kb, t)
    y2 = feynman_48(n, kb, t, v1, v2)
    y = jnp.hstack([k.reshape(-1, 1) for k in [y0, y1, y2]])
    jnp.save(f"{file_path}_x.npy", X)
    jnp.save(f"{file_path}_y.npy", y)


def feynman_13_61_62_70_77_100():
    file_path = "feynman_13_61_62_70_77_100"
    min_vals = jnp.array([1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 3., 1., 1., 1., 1.])
    max_vals = jnp.array([3., 3., 5., 5., 3., 3., 3., 3., 3., 3., 5., 2., 5., 5., 5.])
    key = jax.random.PRNGKey(0)
    X = jax.random.uniform(key, shape=(500, 15)) * (max_vals - min_vals) + min_vals
    q = X[:, 0]
    ef = X[:, 1]
    b = X[:, 2]
    v = X[:, 3]
    theta = X[:, 4]
    n0 = X[:, 5]
    kb = X[:, 6]
    t = X[:, 7]
    pd = X[:, 8]
    m = X[:, 9]
    omega_0 = X[:, 10]
    omega = X[:, 11]
    g = X[:, 12]
    rho = X[:, 13]
    a = X[:, 14]
    y0 = feynman_13(q, ef, b, v, theta)
    y1 = feynman_61(q, ef, m, omega_0, omega)
    y2 = feynman_62(n0, pd, ef, theta, kb, t)
    y3 = feynman_70(pd, ef, theta)
    y4 = feynman_77(g, q, b, m)
    y5 = feynman_100(rho, q, a, m)
    y = jnp.hstack([k.reshape(-1, 1) for k in [y0, y1, y2, y3, y4, y5]])
    jnp.save(f"{file_path}_x.npy", X)
    jnp.save(f"{file_path}_y.npy", y)


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
    feynman_13_61_62_70_77_100()
