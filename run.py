import time

import yaml
from cgpax.jax_functions import JaxFunction

import cgpax
import wandb
from jax import jit, default_backend
import jax.numpy as jnp
from jax import random

from functools import partial

from cgpax.jax_individual import generate_population

from cgpax.run_utils import __update_config_with_env_data__, __compile_selection__, __compile_mutation__, \
    __init_environment__, __compute_parallel_runs_indexes__, __init_environments__, __compute_masks__, \
    __compile_genome_evaluation__, __init_tracking__, __update_tracking__


def run_merge(config: dict, wdb_run):
    rnd_key = random.PRNGKey(config["seed"])

    if config.get("n_parallel_runs", 1) > 1:
        runs_indexes = __compute_parallel_runs_indexes__(config["n_individuals"], config["n_parallel_runs"])
        config["runs_indexes"] = runs_indexes

    # incremental episode duration
    if config["problem"]["incremental_steps"] > 1:
        envs = __init_environments__(config)
        start_gens = [e["start_gen"] for e in envs]
        env_dict = envs[0]
        env = env_dict["env"]
        fitness_scaler = env_dict["fitness_scaler"]
    else:
        env = __init_environment__(config)
        fitness_scaler = 1.0
    __update_config_with_env_data__(config, env)

    # preliminary evo steps
    genome_mask, mutation_mask = __compute_masks__(config)

    # compilation of functions
    evaluate_genomes = __compile_genome_evaluation__(config, env, config["problem"]["episode_length"])
    select_parents = __compile_selection__(config)
    mutate_genomes = __compile_mutation__(config, genome_mask, mutation_mask)
    replace_invalid_nan_fitness = jit(partial(jnp.nan_to_num, nan=config["nan_replacement"]))

    # initialize tracking
    tracking_objects = __init_tracking__(config)

    rnd_key, genome_key = random.split(rnd_key, 2)
    genomes = generate_population(pop_size=config["n_individuals"] * config.get("n_parallel_runs", 1),
                                  genome_mask=genome_mask, rnd_key=genome_key)

    times = {}

    # evolutionary loop
    for _generation in range(config["n_generations"]):
        # check if env needs update
        if config["problem"]["incremental_steps"] > 1 and _generation in start_gens:
            env_idx = start_gens.index(_generation)
            env_dict = envs[env_idx]
            assert env_dict["start_gen"] == _generation
            env = env_dict["env"]
            fitness_scaler = env_dict["fitness_scaler"]
            print(env_dict["duration"])
            evaluate_genomes = __compile_genome_evaluation__(config, env, env_dict["duration"])
            print("compiled")

        # evaluate population
        rnd_key, *eval_keys = random.split(rnd_key, len(genomes) + 1)
        start_eval = time.process_time()
        fitness_values = replace_invalid_nan_fitness(evaluate_genomes(genomes, jnp.array(eval_keys))) * fitness_scaler
        end_eval = time.process_time()
        times["evaluation_time"] = end_eval - start_eval

        # if multiple evals, need median
        if config["n_evals_per_individual"] > 1:
            fitness_variance = jnp.var(fitness_values, axis=1)
            fitness_values = jnp.median(fitness_values, axis=1)

        else:
            fitness_variance = jnp.zeros(len(fitness_values))

        # TODO this requires double check for the parallel runs!
        # select parents
        rnd_key, select_key = random.split(rnd_key, 2)
        start_selection = time.process_time()
        parents = select_parents(genomes, fitness_values, select_key)
        end_selection = time.process_time()
        times["selection_time"] = end_selection - start_selection

        # compute offspring
        rnd_key, mutate_key = random.split(rnd_key, 2)
        mutate_keys = random.split(mutate_key, len(parents))
        start_offspring = time.process_time()
        new_genomes_matrix = mutate_genomes(parents, mutate_keys)
        new_genomes = jnp.reshape(new_genomes_matrix, (-1, new_genomes_matrix.shape[-1]))
        end_offspring = time.process_time()
        times["mutation_time"] = end_offspring - start_offspring

        # print progress
        print(
            f"{_generation} \t"
            f"E: {times['evaluation_time']:.2f} \t"
            f"S: {times['selection_time']:.2f} \t"
            f"M: {times['mutation_time']:.2f} \t"
            f"FITNESS: {jnp.max(fitness_values)}"
        )

        __update_tracking__(config, tracking_objects, genomes, fitness_values, times, wdb_run)

        # select survivals
        # if config["survival"] == "parents":
        survivals = parents

        # update population
        genomes = jnp.concatenate((survivals, new_genomes))

    return "done"


if __name__ == '__main__':
    assert default_backend() == "gpu"

    config_file = "configs/cgp.yaml"
    config = cgpax.get_config(config_file)
    wdb_run = wandb.init(config=config, project="cgpax")
    run_merge(config, wdb_run)
    wdb_run.finish()
