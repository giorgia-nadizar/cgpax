from functools import partial
from typing import Callable, Dict

from environments.locomotion_wrappers import XYPositionWrapper

from cgpax.jax_encoding import genome_to_cgp_program, genome_to_lgp_program

from jax import lax, jit, vmap
from jax import random
from jax import numpy as jnp


def __evaluate_genome_n_times__(evaluation_function: Callable, genome: jnp.ndarray, rnd_key: random.PRNGKey,
                                config: dict, env: XYPositionWrapper, n_times: int, episode_length: int = 1000) -> Dict:
    rnd_key, *subkeys = random.split(rnd_key, n_times + 1)
    subkeys_array = jnp.array(subkeys)
    partial_evaluate_genome = partial(evaluation_function, config=config, env=env, episode_length=episode_length)
    vmap_evaluate_genome = vmap(partial_evaluate_genome, in_axes=(None, 0))
    return vmap_evaluate_genome(genome, subkeys_array)


def evaluate_cgp_genome_n_times(genome: jnp.ndarray, rnd_key: random.PRNGKey, config: dict, env: XYPositionWrapper,
                                n_times: int, episode_length: int = 1000) -> Dict:
    return __evaluate_genome_n_times__(evaluate_cgp_genome, genome, rnd_key, config, env, n_times, episode_length)


def evaluate_lgp_genome_n_times(genome: jnp.ndarray, rnd_key: random.PRNGKey, config: dict, env: XYPositionWrapper,
                                n_times: int, episode_length: int = 1000) -> Dict:
    return __evaluate_genome_n_times__(evaluate_lgp_genome, genome, rnd_key, config, env, n_times, episode_length)


def __evaluate_program__(program: Callable, program_state_size: int, rnd_key: random.PRNGKey, env: XYPositionWrapper,
                         episode_length: int = 1000) -> Dict:
    state = jit(env.reset)(rnd_key)

    def rollout_loop(carry, x):
        env_state, program_state, cum_reward = carry
        inputs = env_state.obs
        new_program_state, actions = program(inputs, program_state)
        new_state = jit(env.step)(env_state, actions)
        corrected_reward = new_state.reward * (1 - new_state.done)
        new_carry = new_state, new_program_state, cum_reward + corrected_reward
        return new_carry, corrected_reward

    carry, _ = lax.scan(
        f=rollout_loop,
        init=(state, jnp.zeros(program_state_size), state.reward),
        xs=None,
        length=episode_length,
    )

    final_env_state, program_state, fitness = carry
    state_descriptor = final_env_state.info.get("state_descriptor", None)
    return {
        "fitness": fitness,
        "state_descriptor": state_descriptor
    }


def evaluate_cgp_genome(genome: jnp.ndarray, rnd_key: random.PRNGKey, config: dict, env: XYPositionWrapper,
                        episode_length: int = 1000) -> Dict:
    return __evaluate_program__(genome_to_cgp_program(genome, config), config["buffer_size"], rnd_key, env,
                                episode_length)


def evaluate_lgp_genome(genome: jnp.ndarray, rnd_key: random.PRNGKey, config: dict, env: XYPositionWrapper,
                        episode_length: int = 1000) -> Dict:
    return __evaluate_program__(genome_to_lgp_program(genome, config), config["n_registers"], rnd_key, env,
                                episode_length)


def evaluate_cgp_genome_regression(genome: jnp.ndarray, config: dict, observations: jnp.ndarray,
                                   targets: jnp.ndarray) -> float:
    def predict_row(observation: jnp.ndarray, genome: jnp.ndarray, config: dict) -> jnp.ndarray:
        program = genome_to_cgp_program(genome, config)
        _, y_tilde = program(observation, jnp.zeros(config["buffer_size"]))
        return y_tilde

    partial_predict = partial(predict_row, genome=genome, config=config)
    vmap_predict = vmap(partial_predict)
    predicted_targets = vmap_predict(observations)
    predicted_targets = jnp.reshape(predicted_targets, targets.shape)
    print(predicted_targets)
    print(targets)
    delta_squared = jnp.square(targets - predicted_targets)
    total_delta = jnp.sum(delta_squared)
    mse = total_delta / targets.size
    return jnp.sqrt(mse)
