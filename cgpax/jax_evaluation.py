from functools import partial

from cgpax.jax_encoding import genome_to_program

from jax import lax, jit, vmap
from jax import random
from jax import numpy as jnp


def evaluate_genome_n_times(genome: jnp.ndarray, rnd_key: random.PRNGKey, config: dict, env,
                            n_times: int, episode_length: int = 1000) -> jnp.ndarray:
    rnd_key, *subkeys = random.split(rnd_key, n_times + 1)
    subkeys_array = jnp.array(subkeys)
    partial_evaluate_genome = partial(evaluate_genome, config=config, env=env, episode_length=episode_length)
    vmap_evaluate_genome = vmap(partial_evaluate_genome, in_axes=(None, 0))
    return vmap_evaluate_genome(genome, subkeys_array)


def evaluate_genome(genome: jnp.ndarray, rnd_key: random.PRNGKey, config: dict, env,
                    episode_length: int = 1000) -> float:
    program = genome_to_program(genome, config)
    state = jit(env.reset)(rnd_key)

    def rollout_loop(carry, x):
        env_state, buffer, cum_reward = carry
        inputs = env_state.obs
        new_buffer, actions = program(inputs, buffer)
        new_state = jit(env.step)(env_state, actions)
        corrected_reward = new_state.reward * (1 - new_state.done)
        new_carry = new_state, new_buffer, cum_reward + corrected_reward
        return new_carry, corrected_reward

    carry, _ = lax.scan(
        f=rollout_loop,
        init=(state, jnp.zeros(config["buffer_size"]), state.reward),
        xs=None,
        length=episode_length,
    )

    return carry[-1]


def evaluate_genome_regression(genome: jnp.ndarray, config: dict, observations: jnp.ndarray,
                               targets: jnp.ndarray) -> float:
    def predict_row(observation: jnp.ndarray, genome: jnp.ndarray, config: dict) -> jnp.ndarray:
        program = genome_to_program(genome, config)
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
