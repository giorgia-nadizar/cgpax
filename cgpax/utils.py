import numpy as np
import jax.numpy as jnp

from cgpax.jax_functions import JaxFunction


def compute_active(x_genes: jnp.ndarray, y_genes: jnp.ndarray, f_genes: jnp.ndarray, out_genes: jnp.ndarray,
                   config: dict) -> np.ndarray:
    n_nodes = config["n_nodes"]
    n_in = config["n_in"]
    active = np.zeros(n_in + n_nodes)
    active[:n_in] = True
    for i in out_genes:
        __compute_active__(active, x_genes, y_genes, f_genes, n_in, i)
    return active


def __compute_active__(active, x_genes, y_genes, f_genes, n_in, idx):
    if not active[idx]:
        active[idx] = True
        __compute_active__(active, x_genes, y_genes, f_genes, n_in, x_genes[idx - n_in])
        # TODO check index of f_genes
        _, arity = list(JaxFunction.arities.items())[f_genes[idx - n_in]]
        if arity > 1:
            __compute_active__(active, x_genes, y_genes, f_genes, n_in, y_genes[idx - n_in])


def readable_program_from_genome(genome: jnp.ndarray, config: dict):
    n_in = config["n_in"]
    n_nodes = config["n_nodes"]
    x_genes, y_genes, f_genes, out_genes = jnp.split(genome, jnp.asarray([n_nodes, 2 * n_nodes, 3 * n_nodes]))
    active = compute_active(x_genes, y_genes, f_genes, out_genes, config)
    function_names = list(JaxFunction.existing_functions.keys())
    text_function = f"def program(inputs, buffer):\n" \
                    f"  buffer[{list(range(n_in))}] = inputs\n"

    # execution
    for buffer_idx in range(n_in, len(active)):
        if active[buffer_idx]:
            idx = buffer_idx - n_in
            function_name = function_names[f_genes[idx]]
            text_function += f"  buffer[{buffer_idx}] = {function_name}(buffer[{x_genes[idx]}]"
            if JaxFunction.arities[function_name] > 1:
                text_function += f", buffer[{y_genes[idx]}]"
            text_function += ")\n"

    # output selection
    text_function += f"  outputs = buffer[{out_genes}]\n"
    return text_function
