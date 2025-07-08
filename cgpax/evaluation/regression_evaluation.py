from typing import Callable, Dict
import jax.numpy as jnp

from jax import vmap

from cgpax.standard.encoding import genome_to_cgp_program, genome_to_lgp_program, _update_buffer_numeric, \
    _update_register_numeric
from cgpax.utils import identity


def _evaluate_regression_program(program: Callable, program_state_size: int, x_values: jnp.ndarray,
                                 y_values: jnp.ndarray) -> Dict:
    def _compute_prediction(x: jnp.ndarray) -> jnp.ndarray:
        program_state = jnp.zeros(program_state_size, dtype=x.dtype)
        _, prediction = program(x, program_state)
        return prediction

    predictions = vmap(_compute_prediction)(x_values)
    squared_errors = (predictions - y_values) ** 2
    mse = jnp.mean(squared_errors)
    ss_res = jnp.sum((y_values - predictions) ** 2, axis=0)
    ss_tot = jnp.sum((y_values - jnp.mean(y_values, axis=0)) ** 2, axis=0)

    r2 = 1 - ss_res / ss_tot
    return {"error": mse, "r2": jnp.mean(r2)}


def evaluate_cgp_genome(genome: jnp.ndarray, config: Dict, x_values: jnp.ndarray, y_values: jnp.ndarray,
                        inner_evaluator: Callable = _evaluate_regression_program,
                        genome_encoder: Callable = genome_to_cgp_program) -> Dict:
    return inner_evaluator(
        genome_encoder(genome, config, outputs_wrapper=identity, buffer_update_fn=_update_buffer_numeric),
        config["buffer_size"], x_values, y_values)


def evaluate_lgp_genome(genome: jnp.ndarray, config: Dict, x_values: jnp.ndarray, y_values: jnp.ndarray,
                        inner_evaluator: Callable = _evaluate_regression_program,
                        genome_encoder: Callable = genome_to_lgp_program) -> Dict:
    return inner_evaluator(
        genome_encoder(genome, config, outputs_wrapper=identity, register_update_fn=_update_register_numeric),
        config["n_registers"], x_values, y_values)
