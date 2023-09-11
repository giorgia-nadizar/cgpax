from functools import partial
from typing import Callable, Dict, Tuple, Any

from brax.v1.envs import State
from brax.v1.envs.wrappers import EpisodeWrapper

from cgpax.jax_encoding import genome_to_cgp_program, genome_to_lgp_program

from jax import lax, jit, vmap
from jax import random
from jax import numpy as jnp


def __init_rewards_carry__(episode_length: int) -> Tuple[int, int, int]:
    return 0, 0, 0


def __update_rewards_carry__(rew_carry: Tuple[int, int, int], state: State,
                             active_episode: int) -> Tuple[int, int, int]:
    rew_forward, rew_ctrl, rew_healthy = rew_carry
    rew_forward += state.metrics.get("reward_forward", state.metrics.get("reward_run", 0)) * active_episode
    rew_ctrl += state.metrics.get("reward_ctrl", 0) * active_episode
    rew_healthy += state.metrics.get("reward_healthy", state.metrics.get("reward_survive", 0)) * active_episode
    return rew_forward, rew_ctrl, rew_healthy


def __extract_final_rewards_carry__(rew_carry: Tuple[int, int, int]) -> Dict:
    rs_forward, rs_ctrl, rs_healthy = rew_carry
    return {
        "cum_healthy_reward": rs_healthy,
        "cum_ctrl_reward": rs_ctrl,
        "cum_forward_reward": rs_forward,
    }


def __init_detailed_rewards_carry__(episode_length: int) -> Tuple[
    jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, int]:
    return (
        jnp.zeros(episode_length), jnp.zeros(episode_length), jnp.zeros(episode_length), jnp.zeros(episode_length), 0)


def __update_detailed_rewards_carry__(rew_carry: Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, int],
                                      state: State, active_episode: int) -> Tuple[
    jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, int]:
    rew_forward, rew_ctrl, rew_healthy, rew_tot, i = rew_carry
    rew_forward = rew_forward.at[i].set(
        state.metrics.get("reward_forward", state.metrics.get("reward_run", 0)) * active_episode)
    rew_ctrl = rew_ctrl.at[i].set(state.metrics.get("reward_ctrl", 0) * active_episode)
    rew_healthy = rew_healthy.at[i].set(
        state.metrics.get("reward_healthy", state.metrics.get("reward_survive", 0)) * active_episode)
    rew_tot = rew_tot.at[i].set(state.reward * active_episode)
    return rew_forward, rew_ctrl, rew_healthy, rew_tot, i + 1


def __extract_final_detailed_rewards_carry__(
        rew_carry: Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, int]) -> Dict:
    rs_forward, rs_ctrl, rs_healthy, rs_total, _ = rew_carry
    return {
        "cum_healthy_reward": jnp.sum(rs_healthy),
        "cum_ctrl_reward": jnp.sum(rs_ctrl),
        "cum_forward_reward": jnp.sum(rs_forward),
        "healthy_rewards": rs_healthy,
        "ctrl_rewards": rs_ctrl,
        "forward_rewards": rs_forward,
        "total_rewards": rs_total,
    }


def __evaluate_program_global__(program: Callable, program_state_size: int, rnd_key: random.PRNGKey,
                                env: EpisodeWrapper, episode_length: int = 1000,
                                carry_init: Callable = __init_rewards_carry__,
                                carry_update: Callable = __update_rewards_carry__,
                                carry_extractor: Callable = __extract_final_rewards_carry__) -> Dict:
    initial_env_state = jit(env.reset)(rnd_key)
    initial_rewards_carry = carry_init(episode_length)

    def rollout_loop(carry: Tuple[State, jnp.ndarray, float, Tuple[int, int, int], int],
                     unused_arg: Any) -> Tuple[Tuple[State, jnp.ndarray, float, Tuple[int, int, int], int], Any]:
        env_state, program_state, cum_reward, rew_carry, active_episode = carry
        inputs = env_state.obs
        new_program_state, actions = program(inputs, program_state)
        new_state = jit(env.step)(env_state, actions)
        corrected_reward = new_state.reward * active_episode
        new_rew_carry = carry_update(rew_carry, new_state, active_episode)
        new_active_episode = (active_episode * (1 - new_state.done)).astype(int)
        new_carry = new_state, new_program_state, cum_reward + corrected_reward, new_rew_carry, new_active_episode
        return new_carry, corrected_reward

    (final_env_state, _, cum_reward, reward_carry, _), _ = lax.scan(
        f=rollout_loop,
        init=(initial_env_state, jnp.zeros(program_state_size), initial_env_state.reward, initial_rewards_carry, 1),
        xs=(),
        length=episode_length,
    )
    x_distance = final_env_state.metrics.get("x_position", 0) - initial_env_state.metrics.get("x_position", 0)
    rewards_dictionary = carry_extractor(reward_carry)
    rewards_dictionary["cum_reward"] = cum_reward
    rewards_dictionary["x_distance"] = x_distance
    return rewards_dictionary


def __evaluate_program__(program: Callable, program_state_size: int, rnd_key: random.PRNGKey, env: EpisodeWrapper,
                         episode_length: int = 1000) -> Dict:
    return __evaluate_program_global__(program, program_state_size, rnd_key, env, episode_length,
                                       __init_rewards_carry__, __update_rewards_carry__,
                                       __extract_final_rewards_carry__)


def __evaluate_program_detailed_tracking__(program: Callable, program_state_size: int, rnd_key: random.PRNGKey,
                                           env: EpisodeWrapper, episode_length: int = 1000) -> Dict:
    return __evaluate_program_global__(program, program_state_size, rnd_key, env, episode_length,
                                       __init_detailed_rewards_carry__, __update_detailed_rewards_carry__,
                                       __extract_final_detailed_rewards_carry__)


def __evaluate_genome_n_times__(evaluation_function: Callable, genome: jnp.ndarray, rnd_key: random.PRNGKey,
                                config: dict, env: EpisodeWrapper, n_times: int,
                                episode_length: int = 1000, inner_evaluator: Callable = __evaluate_program__) -> Dict:
    rnd_key, *subkeys = random.split(rnd_key, n_times + 1)
    subkeys_array = jnp.array(subkeys)
    partial_evaluate_genome = partial(evaluation_function, config=config, env=env, episode_length=episode_length,
                                      inner_evaluator=inner_evaluator)
    vmap_evaluate_genome = vmap(partial_evaluate_genome, in_axes=(None, 0))
    return vmap_evaluate_genome(genome, subkeys_array)


def evaluate_cgp_genome_n_times(genome: jnp.ndarray, rnd_key: random.PRNGKey, config: dict,
                                env: EpisodeWrapper, n_times: int, episode_length: int = 1000,
                                inner_evaluator: Callable = __evaluate_program__) -> Dict:
    return __evaluate_genome_n_times__(evaluate_cgp_genome, genome, rnd_key, config, env, n_times, episode_length,
                                       inner_evaluator=inner_evaluator)


def evaluate_lgp_genome_n_times(genome: jnp.ndarray, rnd_key: random.PRNGKey, config: dict,
                                env: EpisodeWrapper, n_times: int, episode_length: int = 1000,
                                inner_evaluator: Callable = __evaluate_program__) -> Dict:
    return __evaluate_genome_n_times__(evaluate_lgp_genome, genome, rnd_key, config, env, n_times, episode_length,
                                       inner_evaluator=inner_evaluator)


def evaluate_cgp_genome(genome: jnp.ndarray, rnd_key: random.PRNGKey, config: dict, env: EpisodeWrapper,
                        episode_length: int = 1000, inner_evaluator: Callable = __evaluate_program__) -> Dict:
    return inner_evaluator(genome_to_cgp_program(genome, config), config["buffer_size"], rnd_key, env,
                           episode_length)


def evaluate_lgp_genome(genome: jnp.ndarray, rnd_key: random.PRNGKey, config: dict, env: EpisodeWrapper,
                        episode_length: int = 1000, inner_evaluator: Callable = __evaluate_program__) -> Dict:
    return inner_evaluator(genome_to_lgp_program(genome, config), config["n_registers"], rnd_key, env,
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

# TODO explore QD tracking
# def __evaluate_program__(program: Callable, program_state_size: int, rnd_key: random.PRNGKey,
#                          env: EpisodeWrapper, episode_length: int = 1000) -> Dict:
#     initial_env_state = jit(env.reset)(rnd_key)
#
#     def rollout_loop(carry: Tuple[State, jnp.ndarray, float],
#                      unused_arg: Any) -> Tuple[Tuple[State, jnp.ndarray, float], Transition]:
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
