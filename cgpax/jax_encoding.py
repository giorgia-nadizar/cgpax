import jax.numpy as jnp
from jax import jit
from jax.lax import fori_loop

from cgpax.jax_functions import function_switch


@jit
def __copy_inputs__(idx, carry):
    inputs, buffer = carry
    buffer = buffer.at[idx].set(inputs.at[idx].get())
    return inputs, buffer


@jit
def __copy_outputs__(out_idx, carry):
    out_genes, buffer, outputs = carry
    outputs = outputs.at[out_idx].set(
        buffer.at[out_genes.at[out_idx].get()].get()
    )
    return out_genes, buffer, outputs


@jit
def __update_buffer__(buffer_idx, carry):
    x_genes, y_genes, f_genes, buffer = carry
    n_in = len(buffer) - len(x_genes)
    idx = buffer_idx - n_in
    f_idx = f_genes.at[idx].get()
    x_arg = buffer.at[x_genes.at[idx].get()].get()
    y_arg = buffer.at[y_genes.at[idx].get()].get()

    buffer = buffer.at[buffer_idx].set(function_switch(f_idx, x_arg, y_arg))
    return x_genes, y_genes, f_genes, buffer


# @jit
# def __update_buffer_active_only__(carry, buffer_idx):
#     carry = __update_buffer__(buffer_idx, carry)
#     return carry, None


def genome_to_program(genome: jnp.ndarray, config: dict):
    n_in = config["n_in"]
    n_nodes = config["n_nodes"]

    x_genes, y_genes, f_genes, out_genes = jnp.split(genome, [n_nodes, 2 * n_nodes, 3 * n_nodes])

    def program(inputs: jnp.ndarray, buffer: jnp.ndarray) -> (jnp.ndarray, jnp.ndarray):
        _, buffer = fori_loop(0, n_in, __copy_inputs__, (inputs, buffer))
        _, _, _, buffer = fori_loop(n_in, len(buffer), __update_buffer__, (x_genes, y_genes, f_genes, buffer))
        # _, _, outputs = fori_loop(0, n_out, __copy_outputs__, (out_genes, buffer, jnp.zeros(n_out)))
        outputs = jnp.take(buffer, out_genes)
        # TODO add the flag for non-bounded outputs
        bounded_outputs = jnp.tanh(outputs)

        return buffer, bounded_outputs

    return jit(program)

# def genome_and_mask_to_program(genome: jnp.ndarray, config: dict):
#     n_in = config["n_in"]
#     n_nodes = config["n_nodes"]
#     n_out = config["n_out"]
#
#     x_genes, y_genes, f_genes, out_genes, mask = jnp.split(genome,
#                                                            [n_nodes, 2 * n_nodes, 3 * n_nodes, 3 * n_nodes + n_out])
#     actives = jnp.where((mask > 0), size=len(mask), fill_value=jnp.nan)[0]
#     _, actives = jnp.split(actives, [n_in])
#     actives = actives.astype(int)
#
#     def program(inputs: jnp.ndarray, buffer: jnp.ndarray) -> (jnp.ndarray, jnp.ndarray):
#         _, buffer = fori_loop(0, n_in, __copy_inputs__, (inputs, buffer))
#         (_, _, _, buffer), _ = scan(__update_buffer_active_only__, (x_genes, y_genes, f_genes, buffer), actives)
#         _, _, outputs = fori_loop(0, n_out, __copy_outputs__, (out_genes, buffer, jnp.zeros(n_out)))
#         return buffer, outputs
#
#     return jit(program)