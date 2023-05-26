import time

from jax import jit, vmap, default_backend
import jax.numpy as jnp
from jax import random
from brax import envs
from brax.envs.wrapper import EpisodeWrapper

from functools import partial

import wandb

from cgpax.jax_evaluation import evaluate_genome, evaluate_genome_n_times
from cgpax.jax_functions import JaxFunction
from cgpax.jax_individual import compute_genome_mask, compute_mutation_prob_mask, generate_population, \
    mutate_genome_n_times
from cgpax.jax_selection import fp_selection, tournament_selection, island_selection
from cgpax.jax_tracker import Tracker

# TODO read this from file
config = {
    "seed": 0,
    "problem": {"environment": "halfcheetah", "maximize": True, "episode_length": 1000},
    "backend": "positional",
    "n_generations": 5000,
    "n_individuals": 50,
    "selection": {
        "type": "island",
        "elite_size": 10,
        # "tour_size": 3,
    },
    "p_mut_inputs": 0.1,
    "p_mut_functions": 0.1,
    "p_mut_outputs": 0.3,

    "n_evals_per_individual": 10,

    "n_nodes": 50,
    "n_functions": len(JaxFunction.existing_functions),
    "recursive": True,

    "constrain_outputs": True,

    "nan_replacement": -10e04
}


def run(config: dict, wdb_run):
    rnd_key = random.PRNGKey(config["seed"])
    env = envs.get_environment(env_name=config["problem"]["environment"], backend=config["backend"])
    env = EpisodeWrapper(
        env, episode_length=config["problem"]["episode_length"], action_repeat=1
    )

    config["n_in"] = env.observation_size
    config["n_out"] = env.action_size
    config["buffer_size"] = config["n_in"] + config["n_nodes"]
    config["genome_size"] = 3 * config["n_nodes"] + config["n_out"]
    n_mutations_per_individual = int(
        (config["n_individuals"] - config["selection"]["elite_size"]) / config["selection"]["elite_size"])
    nan_replacement = config["nan_replacement"]

    # preliminary evo steps
    genome_mask = compute_genome_mask(config, config["n_in"], config["n_out"])
    mutation_mask = compute_mutation_prob_mask(config, config["n_out"])

    # compilation of functions
    # evaluation
    if config["n_evals_per_individual"] == 1:
        partial_eval_genome = partial(evaluate_genome, config=config, env=env)
    else:
        partial_eval_genome = partial(evaluate_genome_n_times, config=config, env=env,
                                      n_times=config["n_evals_per_individual"])
    vmap_evaluate_genome = vmap(partial_eval_genome, in_axes=(0, 0))
    jit_vmap_evaluate_genome = jit(vmap_evaluate_genome)

    # selection
    if config["selection"]["type"] == "tournament":
        partial_selection = partial(tournament_selection, n_elites=config["selection"]["elite_size"],
                                    tour_size=config["tour_size"])
    elif config["selection"]["type"] == "fp":
        partial_selection = partial(fp_selection, n_elites=config["selection"]["elite_size"])
    else:
        partial_selection = partial(island_selection, n_elites=config["selection"]["elite_size"])
    jit_partial_selection = jit(partial_selection)
    # mutation
    partial_multiple_mutations = partial(mutate_genome_n_times, n_mutations=n_mutations_per_individual,
                                         genome_mask=genome_mask, mutation_mask=mutation_mask)
    vmap_multiple_mutations = vmap(partial_multiple_mutations)
    jit_vmap_multiple_mutations = jit(vmap_multiple_mutations)
    # replace invalid fitness values
    fitness_replacement = jit(partial(jnp.nan_to_num, nan=nan_replacement))

    tracker = Tracker(config)
    tracker_state = tracker.init()

    # initialization
    rnd_key, genome_key = random.split(rnd_key, 2)
    genomes = generate_population(pop_size=config["n_individuals"], genome_mask=genome_mask, rnd_key=genome_key)

    # evolutionary loop
    for _generation in range(config["n_generations"]):
        # evaluate population
        rnd_key, *eval_keys = random.split(rnd_key, len(genomes) + 1)
        start_eval = time.process_time()
        eval_keys_array = jnp.array(eval_keys)
        fitness_values = jit_vmap_evaluate_genome(genomes, eval_keys_array)
        fitness_values = fitness_replacement(fitness_values)
        end_eval = time.process_time()
        evaluation_time = end_eval - start_eval

        # if multiple evals, need median
        if config["n_evals_per_individual"] > 1:
            fitness_median = jnp.median(fitness_values, axis=1)
            fitness_variance = jnp.var(fitness_values, axis=1)
        else:
            fitness_median = fitness_values
            fitness_variance = jnp.zeros(len(fitness_median))

        # max index
        best_genome = genomes.at[jnp.argmax(fitness_median)].get()
        best_fitness = jnp.max(fitness_median)

        # select parents
        rnd_key, fp_key = random.split(rnd_key, 2)
        start_selection = time.process_time()
        parents = jit_partial_selection(genomes, fitness_median, fp_key)
        end_selection = time.process_time()
        selection_time = end_selection - start_selection

        # compute offspring
        rnd_key, mutate_key = random.split(rnd_key, 2)
        mutate_keys = random.split(mutate_key, len(parents))
        start_offspring = time.process_time()
        new_genomes_matrix = jit_vmap_multiple_mutations(parents, mutate_keys)
        new_genomes = jnp.reshape(new_genomes_matrix, (-1, new_genomes_matrix.shape[-1]))
        end_offspring = time.process_time()
        mutation_time = end_offspring - start_offspring

        # print progress
        print(
            f"{_generation} \t"
            f"E: {evaluation_time:.2f} \t"
            f"S: {selection_time:.2f} \t"
            f"M: {mutation_time:.2f} \t"
            f"FITNESS: {best_fitness}")

        # update tracker
        tracker_state = tracker.update(
            tracker_state=tracker_state,
            fitness=fitness_median,
            best_individual=best_genome,
            selection_time=selection_time,
            mutation_time=mutation_time,
            evaluation_time=evaluation_time
        )
        tracker.wandb_log(tracker_state, wdb_run)

        # update population
        genomes = jnp.concatenate((parents, new_genomes))

    tracker.wandb_save_genome(best_genome, wdb_run)
    return "done"


if __name__ == "__main__":
    assert default_backend() == "gpu"

    wdb_run = wandb.init(config=config, project="cgpax")

    for seed in range(1):
        config["seed"] = seed
        run(config, wdb_run)

    wdb_run.finish()
