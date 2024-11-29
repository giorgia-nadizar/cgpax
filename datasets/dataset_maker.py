import jax.numpy as jnp


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
    jnp.save(f"datasets/{file_path}_x.npy", two_bits_input())

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
    jnp.save(f"datasets/{file_path}_y.npy", outputs)


def two_bit_multiplier():
    file_path = "two_bit_multiplier"

    jnp.save(f"datasets/{file_path}_x.npy", two_bits_input())

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
    jnp.save(f"datasets/{file_path}_y.npy", outputs)


if __name__ == '__main__':
    two_bit_multiplier()
