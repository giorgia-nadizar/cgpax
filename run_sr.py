import time

from jax import jit, vmap, default_backend
import jax.numpy as jnp
from jax import random
from sklearn import datasets

from functools import partial

import cgpax.utils
import wandb

from cgpax.jax_evaluation import evaluate_cgp_genome_regression
from cgpax.jax_functions import JaxFunction
from cgpax.jax_individual import compute_lgp_genome_mask, compute_cgp_mutation_prob_mask, generate_population, \
    mutate_genome_n_times
from cgpax.jax_selection import fp_selection
from cgpax.jax_tracker import Tracker

# TODO read this from file
config = {
    "seed": 0,
    "problem": "gravity",
    "n_generations": 10000,
    "n_individuals": 15,
    "elite_size": 5,
    "p_mut_inputs": 0.1,
    "p_mut_functions": 0.1,
    "p_mut_outputs": 0.3,

    "n_nodes": 10,
    "n_functions": len(JaxFunction.existing_functions),

    "constrain_outputs": True,

    "nan_replacement": 10e10
}


def gravity(X):
    m1 = X[:, 0]
    m2 = X[:, 1]
    r = X[:, 2]
    return (m1 * m2) / (r * r)


def run(config: dict, wdb_run):
    rnd_key = random.PRNGKey(config["seed"])
    rnd_key, dataset_key = random.split(rnd_key, 2)
    X = random.normal(dataset_key, shape=(1000, 3))
    y = gravity(X)
    observations = X
    targets = y

    config["n_in"] = observations.shape[1]
    config["n_out"] = 1
    config["buffer_size"] = config["n_in"] + config["n_nodes"]
    config["genome_size"] = 3 * config["n_nodes"] + config["n_out"]
    n_mutations_per_individual = int((config["n_individuals"] - config["elite_size"]) / config["elite_size"])
    nan_replacement = config["nan_replacement"]

    # preliminary evo steps
    genome_mask = compute_lgp_genome_mask(config, config["n_in"], config["n_out"])
    mutation_mask = compute_cgp_mutation_prob_mask(config, config["n_out"])

    # compilation of functions
    # evaluation
    partial_eval_genome = partial(evaluate_cgp_genome_regression, config=config, observations=observations,
                                  targets=targets)
    vmap_evaluate_genome = vmap(partial_eval_genome)
    jit_vmap_evaluate_genome = jit(vmap_evaluate_genome)

    # selection
    partial_fp_selection = partial(fp_selection, n_elites=config["elite_size"])
    jit_partial_fp_selection = jit(partial_fp_selection)
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
    times = {}
    for _generation in range(config["n_generations"]):
        # evaluate population
        start_eval = time.process_time()
        fitness_values = jit_vmap_evaluate_genome(genomes)
        fitness_values = fitness_replacement(fitness_values)
        fitness_values = - fitness_values  # we would need to minimize the error
        end_eval = time.process_time()
        times["evaluation_time"] = end_eval - start_eval

        # select parents
        rnd_key, fp_key = random.split(rnd_key, 2)
        start_selection = time.process_time()
        parents = jit_partial_fp_selection(genomes, fitness_values, fp_key)
        end_selection = time.process_time()
        times["selection_time"] = end_selection - start_selection

        # compute offspring
        rnd_key, mutate_key = random.split(rnd_key, 2)
        mutate_keys = random.split(mutate_key, len(parents))
        start_offspring = time.process_time()
        new_genomes_matrix = jit_vmap_multiple_mutations(parents, mutate_keys)
        new_genomes = jnp.reshape(new_genomes_matrix, (-1, new_genomes_matrix.shape[-1]))
        end_offspring = time.process_time()
        times["mutation_time"] = end_offspring - start_offspring

        # max index
        best_genome = genomes.at[jnp.argmax(fitness_values)].get()
        best_fitness = jnp.max(fitness_values)
        best_program = cgpax.utils.readable_cgp_program_from_genome(best_genome, config)

        # print progress
        print(
            f"{_generation} \t"
            f"E: {times['evaluation_time']:.2f} \t"
            f"S: {times['selection_time']:.2f} \t"
            f"M: {times['mutation_time']:.2f} \t"
            f"FITNESS: {best_fitness}"
        )
        print(best_program)

        # update tracker
        tracker_state = tracker.update(
            tracker_state=tracker_state,
            fitness=fitness_values,
            best_individual=best_genome,
            times=times
        )
        tracker.wandb_log(tracker_state, wdb_run)

        # update population
        genomes = jnp.concatenate((parents, new_genomes))

    return "done"


if __name__ == "__main__":
    assert default_backend() == "gpu"

    wdb_run = wandb.init(config=config, project="cgpax")

    for seed in range(3):
        config["seed"] = seed
        run(config, wdb_run)

    wdb_run.finish()
