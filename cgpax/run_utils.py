from functools import partial
from typing import List, Callable, Tuple, Dict, Union, Set

from brax.v1 import envs
from brax.v1.envs.wrappers import EpisodeWrapper
from wandb.apis.public import Run

from qdax.environments.locomotion_wrappers import QDEnv, FeetContactWrapper
from jax import vmap, jit, random
import jax.numpy as jnp

from cgpax.jax_evaluation import evaluate_cgp_genome, evaluate_cgp_genome_n_times, evaluate_lgp_genome, \
    evaluate_lgp_genome_n_times
from cgpax.jax_individual import mutate_genome_n_times, mutate_genome_n_times_stacked, compute_cgp_genome_mask, \
    compute_cgp_mutation_prob_mask, compute_lgp_genome_mask, compute_lgp_mutation_prob_mask, \
    levels_back_transformation_function, lgp_one_point_crossover_genomes
from cgpax.jax_selection import truncation_selection, tournament_selection, fp_selection, composed_selection
from cgpax.jax_tracker import Tracker
from cgpax.utils import identity


def __init_environment__(env_name: str, episode_length: int,
                         terminate_when_unhealthy: bool = True) -> Union[QDEnv, EpisodeWrapper]:
    env = envs.get_environment(env_name=env_name, terminate_when_unhealthy=terminate_when_unhealthy)
    env = EpisodeWrapper(env, episode_length=episode_length, action_repeat=1)
    # env = FeetContactWrapper(env=env, env_name=env_name)
    return env


def __init_environment_from_config__(config: Dict) -> Union[QDEnv, EpisodeWrapper]:
    return __init_environment__(config["problem"]["environment"], config["problem"]["episode_length"],
                                config.get("unhealthy_termination", True))


def __init_environments__(config: Dict) -> List[Dict]:
    n_steps = config["problem"]["incremental_steps"]
    min_duration = config["problem"]["min_length"]
    step_size = (config["problem"]["episode_length"] - min_duration) / (n_steps - 1)
    gen_step_size = int(config["n_generations"] / n_steps)
    return [
        {
            "start_gen": gen_step_size * n,
            "env": __init_environment__(env_name=config["problem"]["environment"],
                                        episode_length=(min_duration + int(step_size * n))
                                        ),
            "fitness_scaler": config["problem"]["episode_length"] / (min_duration + int(step_size * n)),
            "duration": (min_duration + int(step_size * n))
        }
        for n in range(n_steps)
    ]


def __update_config_with_env_data__(config: Dict, env) -> None:
    config["n_in"] = env.observation_size
    config["n_out"] = env.action_size
    if config["solver"] == "cgp":
        config["buffer_size"] = config["n_in"] + config["n_nodes"]
        config["genome_size"] = 3 * config["n_nodes"] + config["n_out"]
        levels_back = config.get("levels_back")
        if levels_back is not None and levels_back < config["n_in"]:
            config["levels_back"] = config["n_in"]
    else:
        config["n_registers"] = config["n_in"] + config["n_extra_registers"] + config["n_out"]
        config["genome_size"] = 4 * config["n_rows"]


def __compute_parallel_runs_indexes__(n_individuals: int, n_parallel_runs: int, n_elites: int = 1) -> jnp.ndarray:
    indexes = jnp.zeros((n_parallel_runs, n_individuals))
    for run_idx in range(n_parallel_runs):
        for elite_idx in range(n_elites):
            indexes = indexes.at[run_idx, elite_idx].set(run_idx * n_elites + elite_idx)
        for ind_idx in range(n_individuals - n_elites):
            indexes = indexes.at[run_idx, ind_idx + n_elites].set(
                n_elites * n_parallel_runs + ind_idx + (n_individuals - n_elites) * run_idx)
    return indexes.astype(int)


def __compile_genome_evaluation__(config: Dict, env: Union[QDEnv, EpisodeWrapper], episode_length: int) -> Callable:
    if config["solver"] == "cgp":
        eval_func, eval_n_times_func = evaluate_cgp_genome, evaluate_cgp_genome_n_times
    else:
        eval_func, eval_n_times_func = evaluate_lgp_genome, evaluate_lgp_genome_n_times
    if config["n_evals_per_individual"] == 1:
        partial_eval_genome = partial(eval_func, config=config, env=env, episode_length=episode_length)
    else:
        partial_eval_genome = partial(eval_n_times_func, config=config, env=env,
                                      n_times=config["n_evals_per_individual"], episode_length=episode_length)
    vmap_evaluate_genome = vmap(partial_eval_genome, in_axes=(0, 0))
    return jit(vmap_evaluate_genome)


def __compile_crossover__(config: Dict) -> Union[Callable, None]:
    if config.get("crossover", False) and config["solver"] == "lgp":
        vmap_crossover = vmap(lgp_one_point_crossover_genomes, in_axes=(0, 0, 0))
        return jit(vmap_crossover)
    else:
        return None


def __compile_mutation__(config: Dict, genome_mask: jnp.ndarray, mutation_mask: jnp.ndarray,
                         genome_transformation_function: Callable[[jnp.ndarray], jnp.ndarray],
                         n_mutations_per_individual: int = 1) -> Callable:
    if config["mutation"] == "standard":
        partial_multiple_mutations = partial(mutate_genome_n_times, n_mutations=n_mutations_per_individual,
                                             genome_mask=genome_mask, mutation_mask=mutation_mask,
                                             genome_transformation_function=genome_transformation_function)
    else:
        partial_multiple_mutations = partial(mutate_genome_n_times_stacked, n_mutations=n_mutations_per_individual,
                                             genome_mask=genome_mask, mutation_mask=mutation_mask,
                                             genome_transformation_function=genome_transformation_function)
    vmap_multiple_mutations = vmap(partial_multiple_mutations)
    return jit(vmap_multiple_mutations)


def __compile_survival_selection__(config: Dict) -> Union[Callable, None]:
    if config["survival"] == "parents":
        return None
    elif config["survival"] == "truncation":
        return jit(partial(truncation_selection, n_elites=config["selection"]["elite_size"]))
    elif config["survival"] == "tournament":
        return jit(partial(tournament_selection, n_elites=config["selection"]["elite_size"],
                           tour_size=config["selection"]["tour_size"]))
    else:
        return jit(partial(fp_selection, n_elites=config["selection"]["elite_size"]))


def __compile_parents_selection__(config: Dict, n_parents: int = 0) -> Callable:
    if n_parents == 0:
        n_parents = config["n_individuals"] - config["selection"]["elite_size"]
    if config["selection"]["type"] == "truncation":
        partial_selection = partial(truncation_selection, n_elites=n_parents)
    elif config["selection"]["type"] == "tournament":
        partial_selection = partial(tournament_selection, n_elites=n_parents,
                                    tour_size=config["selection"]["tour_size"])
    else:
        partial_selection = partial(fp_selection, n_elites=n_parents)
    inner_selection = jit(partial_selection)
    if config.get("n_parallel_runs", 1) == 1:
        return inner_selection
    else:
        def composite_selection(genomes, fitness_values, select_key):
            parents_list = []
            for run_idx in config["runs_indexes"]:
                rnd_key, sel_key = random.split(select_key, 2)
                current_parents = composed_selection(genomes, fitness_values, sel_key, run_idx, inner_selection)
                parents_list.append(current_parents)
            parents_matrix = jnp.array(parents_list)
            return jnp.reshape(parents_matrix, (-1, parents_matrix.shape[-1]))

        return composite_selection


def __compute_masks__(config: Dict) -> Tuple[jnp.ndarray, jnp.ndarray]:
    if config["solver"] == "cgp":
        genome_mask = compute_cgp_genome_mask(config, config["n_in"], config["n_out"])
        mutation_mask = compute_cgp_mutation_prob_mask(config, config["n_out"])
    else:
        genome_mask = compute_lgp_genome_mask(config, config["n_in"])
        mutation_mask = compute_lgp_mutation_prob_mask(config)
    return genome_mask, mutation_mask


def __compute_genome_transformation_function__(config: Dict) -> Callable[[jnp.ndarray], jnp.ndarray]:
    if config["solver"] == "cgp" and config.get("levels_back") is not None:
        return levels_back_transformation_function(config["n_in"], config["n_nodes"])
    else:
        return identity


def __init_tracking__(config: Dict) -> Tuple:
    if config.get("n_parallel_runs", 1) > 1:
        trackers = [Tracker(config, idx=k) for k in range(config["n_parallel_runs"])]
        tracker_states = [t.init() for t in trackers]
        return trackers, tracker_states
    else:
        tracker = Tracker(config, idx=config["seed"])
        tracker_state = tracker.init()
        return tracker, tracker_state


def __update_tracking__(config: Dict, tracking_objects: Tuple, genomes: jnp.ndarray, fitness_values: jnp.ndarray,
                        rewards: jnp.ndarray, detailed_rewards: Dict, times: Dict, wdb_run: Run) -> Tuple:
    if config.get("n_parallel_runs", 1) == 1:
        tracker, tracker_state = tracking_objects
        tracker_state = tracker.update(
            tracker_state=tracker_state,
            fitness=fitness_values,
            rewards=rewards,
            detailed_rewards=detailed_rewards,
            best_individual=genomes.at[jnp.argmax(fitness_values)].get(),
            times=times
        )
        tracker.wandb_log(tracker_state, wdb_run)
        return tracker, tracker_state
    else:
        trackers, tracker_states = tracking_objects
        for run_idx in range(config["n_parallel_runs"]):
            current_indexes = config["runs_indexes"].at[run_idx, :].get()
            sub_fitness = jnp.take(fitness_values, current_indexes, axis=0)
            sub_rewards = jnp.take(rewards, current_indexes, axis=0)
            sub_genomes = jnp.take(genomes, current_indexes, axis=0)
            tracker_states[run_idx] = trackers[run_idx].update(
                tracker_state=tracker_states[run_idx],
                fitness=sub_fitness,
                rewards=sub_rewards,
                detailed_rewards=detailed_rewards,
                best_individual=sub_genomes.at[jnp.argmax(sub_fitness)].get(),
                times=times
            )
            trackers[run_idx].wandb_log(tracker_states[run_idx], wdb_run)
        return trackers, tracker_states


def __normalize_array__(array: jnp.ndarray) -> jnp.ndarray:
    min_val = jnp.min(array)
    max_val = jnp.max(array)
    return (array - min_val) / (max_val - min_val)


@jit
def __compute_max_distance__(x_coord: float, x_pos_archive: jnp.ndarray) -> jnp.ndarray:
    @jit
    def __distance__(x1: float, x2: float) -> float:
        return jnp.abs(x1 - x2)

    distances = vmap(__distance__, in_axes=(None, 0))(x_coord, x_pos_archive)
    return jnp.max(distances)


def __compute_novelty_scores__(final_positions: jnp.ndarray, novelty_archive: Set, decimals: int = 2) -> jnp.ndarray:
    x_coordinates = final_positions.at[:, :, 0].get()
    x_coordinates_flat = x_coordinates.flatten()
    archive_array = jnp.asarray(list(novelty_archive))
    max_distances = vmap(__compute_max_distance__, in_axes=(0, None))(x_coordinates_flat, archive_array)
    max_distances = max_distances.reshape(x_coordinates.shape)
    rounded_positions = jnp.around(x_coordinates_flat, decimals=decimals)
    for pos in rounded_positions[~jnp.isnan(rounded_positions)]:
        novelty_archive.add(float(pos))
    return max_distances
