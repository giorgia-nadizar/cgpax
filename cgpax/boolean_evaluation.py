from typing import Callable, Dict
import jax.numpy as jnp

from jax import vmap

from cgpax.standard.encoding import genome_to_cgp_program, genome_to_lgp_program, _update_buffer_boolean, \
    _update_register_boolean
from cgpax.utils import identity


def _evaluate_boolean_program(program: Callable, program_state_size: int, x_values: jnp.ndarray,
                              y_values: jnp.ndarray) -> Dict:
    def _compute_prediction(x: jnp.ndarray) -> jnp.ndarray:
        program_state = jnp.zeros(program_state_size, dtype=x.dtype)
        _, prediction = program(x, program_state)
        return prediction

    predictions = vmap(_compute_prediction)(x_values)
    errors = jnp.abs(predictions - y_values)
    avg_error = jnp.mean(errors)
    return {"error": avg_error, "accuracy": 1 - avg_error}


def evaluate_cgp_genome(genome: jnp.ndarray, config: Dict, x_values: jnp.ndarray, y_values: jnp.ndarray,
                        inner_evaluator: Callable = _evaluate_boolean_program,
                        genome_encoder: Callable = genome_to_cgp_program) -> Dict:
    return inner_evaluator(
        genome_encoder(genome, config, outputs_wrapper=identity, buffer_update_fn=_update_buffer_boolean),
        config["buffer_size"], x_values, y_values)


def evaluate_lgp_genome(genome: jnp.ndarray, config: Dict, x_values: jnp.ndarray, y_values: jnp.ndarray,
                        inner_evaluator: Callable = _evaluate_boolean_program,
                        genome_encoder: Callable = genome_to_lgp_program) -> Dict:
    return inner_evaluator(
        genome_encoder(genome, config, outputs_wrapper=identity, register_update_fn=_update_register_boolean),
        config["n_registers"], x_values, y_values)
