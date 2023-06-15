from functools import partial
from typing import List, Callable

from brax.v1 import envs
from brax.v1.envs.wrappers import EpisodeWrapper
from jax import vmap, jit, random
import jax.numpy as jnp

from cgpax.jax_evaluation import evaluate_cgp_genome, evaluate_cgp_genome_n_times, evaluate_lgp_genome, \
    evaluate_lgp_genome_n_times
from cgpax.jax_individual import mutate_genome_n_times, mutate_genome_n_times_stacked, compute_cgp_genome_mask, \
    compute_cgp_mutation_prob_mask, compute_lgp_genome_mask, compute_lgp_mutation_prob_mask, \
    levels_back_transformation_function
from cgpax.jax_selection import truncation_selection, tournament_selection, fp_selection, composed_selection
from cgpax.jax_tracker import Tracker
from cgpax.utils import identity


# TODO move within individual files?

def __init_environment__(config: dict) -> EpisodeWrapper:
    env = envs.get_environment(env_name=config["problem"]["environment"])
    return EpisodeWrapper(
        env, episode_length=config["problem"]["episode_length"], action_repeat=1
    )


def __init_environments__(config: dict) -> List:
    n_steps = config["problem"]["incremental_steps"]
    min_duration = config["problem"]["min_length"]
    step_size = (config["problem"]["episode_length"] - min_duration) / (n_steps - 1)
    gen_step_size = int(config["n_generations"] / n_steps)
    return [
        {
            "start_gen": gen_step_size * n,
            "env": EpisodeWrapper(
                envs.get_environment(env_name=config["problem"]["environment"]),
                episode_length=(min_duration + int(step_size * n)), action_repeat=1
            ),
            "fitness_scaler": config["problem"]["episode_length"] / (min_duration + int(step_size * n)),
            "duration": (min_duration + int(step_size * n))
        }
        for n in range(n_steps)
    ]


def __update_config_with_env_data__(config: dict, env) -> None:
    config["n_in"] = env.observation_size
    config["n_out"] = env.action_size
    print(f"{env.observation_size} - {env.action_size}")
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


def __compile_genome_evaluation__(config: dict, env, episode_length: int):
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


def __compile_mutation__(config: dict, genome_mask: jnp.ndarray, mutation_mask: jnp.ndarray,
                         genome_transformation_function: Callable[[jnp.ndarray], jnp.ndarray]):
    n_mutations_per_individual = int(
        (config["n_individuals"] - config["selection"]["elite_size"]) / config["selection"]["elite_size"])
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


def __compile_selection__(config: dict):
    if config["selection"]["type"] == "truncation":
        partial_selection = partial(truncation_selection, n_elites=config["selection"]["elite_size"])
    elif config["selection"]["type"] == "tournament":
        partial_selection = partial(tournament_selection, n_elites=config["selection"]["elite_size"],
                                    tour_size=config["tour_size"])
    else:
        partial_selection = partial(fp_selection, n_elites=config["selection"]["elite_size"])
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


def __compute_masks__(config: dict):
    if config["solver"] == "cgp":
        genome_mask = compute_cgp_genome_mask(config, config["n_in"], config["n_out"])
        mutation_mask = compute_cgp_mutation_prob_mask(config, config["n_out"])
    else:
        genome_mask = compute_lgp_genome_mask(config, config["n_in"])
        mutation_mask = compute_lgp_mutation_prob_mask(config)
    return genome_mask, mutation_mask


def __compute_genome_transformation_function__(config: dict):
    if config["solver"] == "cgp" and config.get("levels_back") is not None:
        return levels_back_transformation_function(config["n_in"], config["n_nodes"])
    else:
        return identity


def __init_tracking__(config: dict):
    if config.get("n_parallel_runs", 1) > 1:
        trackers = [Tracker(config, idx=k) for k in range(config["n_parallel_runs"])]
        tracker_states = [t.init() for t in trackers]
        return trackers, tracker_states
    else:
        tracker = Tracker(config)
        tracker_state = tracker.init()
        return tracker, tracker_state


def __update_tracking__(config: dict, tracking_objects: tuple, genomes: jnp.ndarray, fitness_values: jnp.ndarray,
                        times: dict, wdb_run):
    if config.get("n_parallel_runs", 1) == 1:
        tracker, tracker_state = tracking_objects
        tracker_state = tracker.update(
            tracker_state=tracker_state,
            fitness=fitness_values,
            best_individual=genomes.at[jnp.argmax(fitness_values)].get(),
            times=times
        )
        tracker.wandb_log(tracker_state, wdb_run)
    else:
        trackers, tracker_states = tracking_objects
        for run_idx in range(config["n_parallel_runs"]):
            current_indexes = config["runs_indexes"].at[run_idx, :].get()
            sub_fitness = jnp.take(fitness_values, current_indexes, axis=0)
            sub_genomes = jnp.take(genomes, current_indexes, axis=0)
            tracker_states[run_idx] = trackers[run_idx].update(
                tracker_state=tracker_states[run_idx],
                fitness=sub_fitness,
                best_individual=sub_genomes.at[jnp.argmax(sub_fitness)].get(),
                times=times
            )
            trackers[run_idx].wandb_log(tracker_states[run_idx], wdb_run)
