from typing import Callable, Tuple, Dict

import jax.numpy as jnp
from jax import jit
from jax.lax import fori_loop

from cgpax.functions import function_switch, constants


@jit
def _update_buffer(buffer_idx: int,
                   carry: Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]) -> Tuple[
    jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    x_genes, y_genes, f_genes, buffer = carry
    n_in = len(buffer) - len(x_genes)
    idx = buffer_idx - n_in
    f_idx = f_genes.at[idx].get()
    x_arg = buffer.at[x_genes.at[idx].get()].get()
    y_arg = buffer.at[y_genes.at[idx].get()].get()

    buffer = buffer.at[buffer_idx].set(function_switch(f_idx, x_arg, y_arg))
    return x_genes, y_genes, f_genes, buffer


@jit
def _update_register(row_idx: int, carry: Tuple[
    jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, int, jnp.ndarray]) -> \
        Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, int, jnp.ndarray]:
    lhs_genes, x_genes, y_genes, f_genes, n_in, register = carry
    lhs_idx = lhs_genes.at[row_idx].get() + n_in
    f_idx = f_genes.at[row_idx].get()
    x_idx = x_genes.at[row_idx].get()
    x_arg = register.at[x_idx].get()
    y_idx = y_genes.at[row_idx].get()
    y_arg = register.at[y_idx].get()
    register = register.at[lhs_idx].set(function_switch(f_idx, x_arg, y_arg))
    return lhs_genes, x_genes, y_genes, f_genes, n_in, register


def genome_to_cgp_program(genome: jnp.ndarray, config: Dict,
                          outputs_wrapper: Callable[[jnp.ndarray], jnp.ndarray] = jnp.tanh) -> Callable:
    n_const = config["n_constants"]
    n_in = config["n_in"]
    n_out = config["n_out"]
    n_nodes = config["n_nodes"]

    x_genes, y_genes, f_genes, out_genes = jnp.split(genome, [n_nodes, 2 * n_nodes, 2 * n_nodes + n_out])
    x_genes = x_genes.astype(int)
    y_genes = y_genes.astype(int)
    f_genes = f_genes.astype(int)
    out_genes = out_genes.astype(int)

    def _program(inputs: jnp.ndarray, buffer: jnp.ndarray) -> (jnp.ndarray, jnp.ndarray):
        buffer = jnp.concatenate([inputs, constants[:n_const], buffer[n_in:len(buffer)]])
        _, _, _, buffer = fori_loop(n_in, len(buffer), _update_buffer,
                                       (x_genes, y_genes, f_genes, buffer))
        outputs = jnp.take(buffer, out_genes)
        bounded_outputs = outputs_wrapper(outputs)

        return buffer, bounded_outputs

    return jit(_program)


def genome_to_lgp_program(genome: jnp.ndarray, config: Dict,
                          outputs_wrapper: Callable[[jnp.ndarray], jnp.ndarray] = jnp.tanh) -> Callable:
    n_const = config["n_constants"]
    n_in = config["n_in"]
    n_out = config["n_out"]
    n_rows = config["n_rows"]
    n_registers = config["n_registers"]
    output_positions = jnp.arange(start=n_registers - n_out, stop=n_registers)

    lhs_genes, x_genes, y_genes, f_genes = jnp.split(genome, 4)
    lhs_genes = lhs_genes.astype(int)
    x_genes = x_genes.astype(int)
    y_genes = y_genes.astype(int)
    f_genes = f_genes.astype(int)

    def _program(inputs: jnp.ndarray, register: jnp.ndarray) -> (jnp.ndarray, jnp.ndarray):
        register = jnp.zeros(n_registers)
        register = jnp.concatenate([inputs, constants[:n_const], register[n_in:len(register)]])
        _, _, _, _, _, register = fori_loop(0, n_rows, _update_register,
                                            (lhs_genes, x_genes, y_genes, f_genes, n_in, register))
        outputs = jnp.take(register, output_positions)
        bounded_outputs = outputs_wrapper(outputs)

        return register, bounded_outputs

    return jit(_program)
