from functools import partial

from brax import envs
from brax.envs.wrapper import EpisodeWrapper
from jax import vmap, jit
import jax.numpy as jnp

from cgpax.jax_evaluation import evaluate_cgp_genome, evaluate_cgp_genome_n_times, evaluate_lgp_genome, \
    evaluate_lgp_genome_n_times
from cgpax.jax_individual import mutate_genome_n_times, mutate_genome_n_times_stacked, compute_cgp_genome_mask, \
    compute_cgp_mutation_prob_mask, compute_lgp_genome_mask, compute_lgp_mutation_prob_mask
from cgpax.jax_selection import truncation_selection, tournament_selection, fp_selection


# TODO move within individual files?

def __init_environment__(config: dict):
    env = envs.get_environment(env_name=config["problem"]["environment"], backend=config["backend"])
    return EpisodeWrapper(
        env, episode_length=config["problem"]["episode_length"], action_repeat=1
    )


def __init_environments__(config: dict):
    n_steps = config["problem"]["incremental_steps"]
    min_duration = config["problem"]["min_length"]
    step_size = (config["problem"]["episode_length"] - min_duration) / (n_steps - 1)
    gen_step_size = int(config["n_generations"] / n_steps)
    return [
        {
            "start_gen": gen_step_size * n,
            "env": EpisodeWrapper(
                envs.get_environment(env_name=config["problem"]["environment"], backend=config["backend"]),
                episode_length=(min_duration + int(step_size * n)), action_repeat=1
            ),
            "fitness_scaler": config["problem"]["episode_length"] / (min_duration + int(step_size * n)),
            "duration": (min_duration + int(step_size * n))
        }
        for n in range(n_steps)
    ]


def __update_config_with_env_data__(config: dict, env):
    config["n_in"] = env.observation_size
    config["n_out"] = env.action_size
    if config["solver"] == "cgp":
        config["buffer_size"] = config["n_in"] + config["n_nodes"]
        config["genome_size"] = 3 * config["n_nodes"] + config["n_out"]
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


def __compile_mutation__(config: dict, genome_mask: jnp.ndarray, mutation_mask: jnp.ndarray):
    n_mutations_per_individual = int(
        (config["n_individuals"] - config["selection"]["elite_size"]) / config["selection"]["elite_size"])
    if config["mutation"] == "standard":
        partial_multiple_mutations = partial(mutate_genome_n_times, n_mutations=n_mutations_per_individual,
                                             genome_mask=genome_mask, mutation_mask=mutation_mask)
    else:
        partial_multiple_mutations = partial(mutate_genome_n_times_stacked, n_mutations=n_mutations_per_individual,
                                             genome_mask=genome_mask, mutation_mask=mutation_mask)
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
    return jit(partial_selection)


def __compute_masks__(config: dict):
    if config["solver"] == "cgp":
        genome_mask = compute_cgp_genome_mask(config, config["n_in"], config["n_out"])
        mutation_mask = compute_cgp_mutation_prob_mask(config, config["n_out"])
    else:
        genome_mask = compute_lgp_genome_mask(config, config["n_in"])
        mutation_mask = compute_lgp_mutation_prob_mask(config)
    return genome_mask, mutation_mask
