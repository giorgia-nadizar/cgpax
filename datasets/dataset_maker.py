import jax.numpy as jnp

# 2-bit adder
file_path = "two_bit_adder"
inputs_list = []
for a in [0, 1]:
    for b in [0, 1]:
        for c in [0, 1]:
            for d in [0, 1]:
                inputs_list.append([a, b, c, d])
inputs = jnp.asarray(inputs_list)
jnp.save(f"{file_path}_inputs.npy", inputs)

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
jnp.save(f"{file_path}_outputs.npy", outputs)
