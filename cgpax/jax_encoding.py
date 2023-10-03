from typing import Callable, Tuple, Dict

import jax.numpy as jnp
from jax import jit
from jax.lax import fori_loop

from cgpax.jax_functions import function_switch, constants


@jit
def _offset_copy(idx: int, carry: Tuple[int, jnp.ndarray, jnp.ndarray]) -> Tuple[int, jnp.ndarray, jnp.ndarray]:
    offset, source, buffer = carry
    buffer = buffer.at[idx + offset].set(source.at[idx].get())
    return offset, source, buffer


@jit
def _update_buffer(buffer_idx: int,
                   carry: Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]) -> Tuple[
    jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    x_genes, y_genes, f_genes, weights, buffer = carry
    n_in = len(buffer) - len(x_genes)
    idx = buffer_idx - n_in
    f_idx = f_genes.at[idx].get()
    weight = weights.at[idx].get()
    x_arg = buffer.at[x_genes.at[idx].get()].get()
    y_arg = buffer.at[y_genes.at[idx].get()].get()

    buffer = buffer.at[buffer_idx].set(function_switch(f_idx, x_arg, y_arg) * weight)
    return x_genes, y_genes, f_genes, weights, buffer


@jit
def _update_register(row_idx: int, carry: Tuple[
    jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, int, jnp.ndarray]) -> \
        Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, int, jnp.ndarray]:
    lhs_genes, x_genes, y_genes, f_genes, weights, n_in, register = carry
    lhs_idx = lhs_genes.at[row_idx].get() + n_in
    f_idx = f_genes.at[row_idx].get()
    weight = weights.at[row_idx].get()
    x_idx = x_genes.at[row_idx].get()
    x_arg = register.at[x_idx].get()
    y_idx = y_genes.at[row_idx].get()
    y_arg = register.at[y_idx].get()
    register = register.at[lhs_idx].set(function_switch(f_idx, x_arg, y_arg) * weight)
    return lhs_genes, x_genes, y_genes, f_genes, weights, n_in, register


def genome_to_cgp_program(genome: jnp.ndarray, config: Dict,
                          outputs_wrapper: Callable[[jnp.ndarray], jnp.ndarray] = jnp.tanh) -> Callable:
    n_in_env = config["n_in_env"]
    n_const = config["n_constants"]
    n_in = config["n_in"]
    n_out = config["n_out"]
    n_nodes = config["n_nodes"]

    x_genes, y_genes, f_genes, out_genes, weights = jnp.split(genome,
                                                              [n_nodes, 2 * n_nodes, 3 * n_nodes, 3 * n_nodes + n_out])
    x_genes = x_genes.astype(int)
    y_genes = y_genes.astype(int)
    f_genes = f_genes.astype(int)
    out_genes = out_genes.astype(int)

    def _program(inputs: jnp.ndarray, buffer: jnp.ndarray) -> (jnp.ndarray, jnp.ndarray):
        # copy actual inputs
        _, _, buffer = fori_loop(0, n_in_env, _offset_copy, (0, inputs, buffer))
        # copy constants
        _, _, buffer = fori_loop(0, n_const, _offset_copy, (n_in_env, constants, buffer))

        _, _, _, _, buffer = fori_loop(n_in, len(buffer), _update_buffer,
                                       (x_genes, y_genes, f_genes, weights, buffer))
        outputs = jnp.take(buffer, out_genes)
        bounded_outputs = outputs_wrapper(outputs)

        return buffer, bounded_outputs

    return jit(_program)


def genome_to_lgp_program(genome: jnp.ndarray, config: Dict,
                          outputs_wrapper: Callable[[jnp.ndarray], jnp.ndarray] = jnp.tanh) -> Callable:
    n_in_env = config["n_in_env"]
    n_const = config["n_constants"]
    n_in = config["n_in"]
    n_out = config["n_out"]
    n_rows = config["n_rows"]
    n_registers = config["n_registers"]
    output_positions = jnp.arange(start=n_registers - n_out, stop=n_registers)

    lhs_genes, x_genes, y_genes, f_genes, weights = jnp.split(genome, 5)
    lhs_genes = lhs_genes.astype(int)
    x_genes = x_genes.astype(int)
    y_genes = y_genes.astype(int)
    f_genes = f_genes.astype(int)

    def _program(inputs: jnp.ndarray, register: jnp.ndarray) -> (jnp.ndarray, jnp.ndarray):
        register = jnp.zeros(n_registers)
        # copy actual environment inputs
        _, _, register = fori_loop(0, n_in_env, _offset_copy, (0, inputs, register))
        # copy constant inputs
        _, _, register = fori_loop(0, n_const, _offset_copy, (n_in_env, constants, register))
        _, _, _, _, _, _, register = fori_loop(0, n_rows, _update_register,
                                               (lhs_genes, x_genes, y_genes, f_genes, weights, n_in, register))
        outputs = jnp.take(register, output_positions)
        bounded_outputs = outputs_wrapper(outputs)

        return register, bounded_outputs

    return jit(_program)
