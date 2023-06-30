from functools import partial
from typing import Callable, Dict, Tuple, Any, Union

from brax.v1.envs.wrappers import EpisodeWrapper

from qdax.core.buffer import QDTransition, Transition
from qdax.environments import get_feet_contact_proportion
from qdax.environments.locomotion_wrappers import QDEnv

from cgpax.jax_encoding import genome_to_cgp_program, genome_to_lgp_program

from jax import lax, jit, vmap
from jax import random
from jax import numpy as jnp

from qdax.types import EnvState


def __evaluate_genome_n_times__(evaluation_function: Callable, genome: jnp.ndarray, rnd_key: random.PRNGKey,
                                config: dict, env: Union[QDEnv, EpisodeWrapper], n_times: int,
                                episode_length: int = 1000) -> Dict:
    rnd_key, *subkeys = random.split(rnd_key, n_times + 1)
    subkeys_array = jnp.array(subkeys)
    partial_evaluate_genome = partial(evaluation_function, config=config, env=env, episode_length=episode_length)
    vmap_evaluate_genome = vmap(partial_evaluate_genome, in_axes=(None, 0))
    return vmap_evaluate_genome(genome, subkeys_array)


def evaluate_cgp_genome_n_times(genome: jnp.ndarray, rnd_key: random.PRNGKey, config: dict,
                                env: Union[QDEnv, EpisodeWrapper], n_times: int, episode_length: int = 1000) -> Dict:
    return __evaluate_genome_n_times__(evaluate_cgp_genome, genome, rnd_key, config, env, n_times, episode_length)


def evaluate_lgp_genome_n_times(genome: jnp.ndarray, rnd_key: random.PRNGKey, config: dict,
                                env: Union[QDEnv, EpisodeWrapper], n_times: int, episode_length: int = 1000) -> Dict:
    return __evaluate_genome_n_times__(evaluate_lgp_genome, genome, rnd_key, config, env, n_times, episode_length)


def __evaluate_program__(program: Callable, program_state_size: int, rnd_key: random.PRNGKey,
                         env: Union[QDEnv, EpisodeWrapper], episode_length: int = 1000) -> Dict:
    initial_env_state = jit(env.reset)(rnd_key)
    initial_rewards_carry = (
        jnp.zeros(episode_length), jnp.zeros(episode_length), jnp.zeros(episode_length), jnp.zeros(episode_length), 0)

    def rollout_loop(
            carry: Tuple[
                EnvState, jnp.ndarray, float, Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, int], int],
            unused_arg: Any) -> Tuple[
        Tuple[
            EnvState, jnp.ndarray, float, Tuple[
                jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, int], int], Transition]:
        env_state, program_state, cum_reward, rew_carry, active_episode = carry
        rew_forward, rew_ctrl, rew_healthy, rew_tot, i = rew_carry
        inputs = env_state.obs
        new_program_state, actions = program(inputs, program_state)
        new_state = jit(env.step)(env_state, actions)
        new_active_episode = active_episode * (1 - new_state.done)
        corrected_reward = new_state.reward * new_active_episode
        rew_forward = rew_forward.at[i].set(
            new_state.metrics.get("reward_forward", new_state.metrics.get("reward_run", 0)) * new_active_episode)
        rew_ctrl = rew_ctrl.at[i].set(new_state.metrics.get("reward_ctrl", 0) * new_active_episode)
        rew_healthy = rew_healthy.at[i].set(
            new_state.metrics.get("reward_healthy", new_state.metrics.get("reward_survive", 0)) * new_active_episode)
        rew_tot = rew_tot.at[i].set(new_state.reward * new_active_episode)
        new_rew_carry = rew_forward, rew_ctrl, rew_healthy, rew_tot, i + 1
        new_carry = new_state, new_program_state, cum_reward + corrected_reward, new_rew_carry, new_active_episode
        return new_carry, corrected_reward

    (final_env_state, _, cum_reward, (rs_forward, rs_ctrl, rs_healthy, rs_total, _), _), transitions = lax.scan(
        f=rollout_loop,
        init=(initial_env_state, jnp.zeros(program_state_size), initial_env_state.reward, initial_rewards_carry, 1),
        xs=(),
        length=episode_length,
    )

    x_distance = final_env_state.metrics.get("x_position", 0) - initial_env_state.metrics.get("x_position", 0)

    return {
        "cum_reward": cum_reward,
        "cum_healthy_reward": jnp.sum(rs_healthy),
        "cum_ctrl_reward": jnp.sum(rs_ctrl),
        "cum_forward_reward": jnp.sum(rs_forward),
        "healthy_rewards": rs_healthy,
        "ctrl_rewards": rs_ctrl,
        "forward_rewards": rs_forward,
        "total_rewards": rs_total,
        "x_distance": x_distance,
    }

# TODO explore QD tracking
# def __evaluate_program__(program: Callable, program_state_size: int, rnd_key: random.PRNGKey,
#                          env: Union[QDEnv, EpisodeWrapper], episode_length: int = 1000) -> Dict:
#     initial_env_state = jit(env.reset)(rnd_key)
#
#     def rollout_loop(carry: Tuple[EnvState, jnp.ndarray, float],
#                      unused_arg: Any) -> Tuple[Tuple[EnvState, jnp.ndarray, float], Transition]:
#         env_state, program_state, cum_reward = carry
#         inputs = env_state.obs
#         new_program_state, actions = program(inputs, program_state)
#         new_state = jit(env.step)(env_state, actions)
#         corrected_reward = new_state.reward * (1 - new_state.done)
#         # transition = QDTransition(
#         #     obs=env_state.obs,
#         #     next_obs=new_state.obs,
#         #     rewards=new_state.reward,
#         #     dones=new_state.done,
#         #     actions=actions,
#         #     truncations=new_state.info["truncation"],
#         #     state_desc=env_state.info["state_descriptor"],
#         #     next_state_desc=new_state.info["state_descriptor"],
#         # )
#         new_carry = new_state, new_program_state, cum_reward + corrected_reward
#         # return new_carry, transition
#         return new_carry, corrected_reward
#
#     (final_env_state, _, cum_reward), transitions = lax.scan(
#         f=rollout_loop,
#         init=(initial_env_state, jnp.zeros(program_state_size), initial_env_state.reward),
#         xs=(),
#         length=episode_length,
#     )
#
#     x_distance = final_env_state.metrics.get("x_position", 0) - initial_env_state.metrics.get("x_position", 0)
#     # feet_contact_proportion = get_feet_contact_proportion(transitions)
#     return {
#         "cum_reward": cum_reward,
#         # "feet_contact_proportion": feet_contact_proportion,
#         "x_distance": x_distance
#     }


def evaluate_cgp_genome(genome: jnp.ndarray, rnd_key: random.PRNGKey, config: dict, env: Union[QDEnv, EpisodeWrapper],
                        episode_length: int = 1000) -> Dict:
    return __evaluate_program__(genome_to_cgp_program(genome, config), config["buffer_size"], rnd_key, env,
                                episode_length)


def evaluate_lgp_genome(genome: jnp.ndarray, rnd_key: random.PRNGKey, config: dict, env: Union[QDEnv, EpisodeWrapper],
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
