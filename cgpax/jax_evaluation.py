from cgpax.jax_encoding import genome_to_program

from jax import lax, jit
from jax import random
from jax import numpy as jnp


def evaluate_genome(genome: jnp.ndarray, rnd_key: random.PRNGKey, config: dict, env) -> float:
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
        length=config["problem"]["episode_length"],
    )

    return carry[-1]
