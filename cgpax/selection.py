from functools import partial
from typing import Callable

import jax.numpy as jnp
from jax import random, vmap


def composed_selection(genomes: jnp.ndarray, fitness_values: jnp.ndarray, rnd_key: random.PRNGKey,
                       indexes: jnp.ndarray, selection_function: Callable) -> jnp.ndarray:
    sub_genomes = jnp.take(genomes, indexes, axis=0)
    sub_fitness_values = jnp.take(fitness_values, indexes, axis=0)
    return selection_function(sub_genomes, sub_fitness_values, rnd_key)


def truncation_selection(genomes: jnp.ndarray, fitness_values: jnp.ndarray, rnd_key: random.PRNGKey,
                         n_elites: int) -> jnp.ndarray:
    elites, _ = jnp.split(jnp.argsort(-fitness_values), [n_elites])
    return jnp.take(genomes, elites, axis=0)


def fp_selection(genomes: jnp.ndarray, fitness_values: jnp.ndarray, rnd_key: random.PRNGKey,
                 n_elites: int) -> jnp.ndarray:
    p = 1 - ((jnp.max(fitness_values) - fitness_values) / (jnp.max(fitness_values) - jnp.min(fitness_values)))
    p /= jnp.sum(p)
    return random.choice(rnd_key, genomes, shape=[n_elites], p=p, replace=False)


def tournament_selection(genomes: jnp.ndarray, fitness_values: jnp.ndarray, rnd_key: random.PRNGKey,
                         n_elites: int, tour_size: int) -> jnp.ndarray:
    def _tournament(sample_key: random.PRNGKey, genomes: jnp.ndarray, fitness_values: jnp.ndarray,
                    tour_size: int) -> jnp.ndarray:
        indexes = random.choice(sample_key, jnp.arange(start=0, stop=len(genomes)), shape=[tour_size], replace=True)
        mask = jnp.zeros_like(fitness_values)
        mask = mask.at[indexes].set(1)
        fitness_values_for_selection = (fitness_values + jnp.min(fitness_values) + 1) * mask
        best_genome = genomes.at[jnp.argmax(fitness_values_for_selection)].get()
        return best_genome

    sample_keys = random.split(rnd_key, n_elites)
    partial_single_tournament = partial(_tournament, genomes=genomes, fitness_values=fitness_values,
                                        tour_size=tour_size)
    vmap_tournament = vmap(partial_single_tournament)
    return vmap_tournament(sample_key=sample_keys)


def lexicase_selection(genomes: jnp.ndarray, fitness_values: jnp.ndarray, rnd_key: random.PRNGKey,
                       n_elites: int):
    selected_ids = []

    # Get the number of test cases
    num_cases = fitness_values.shape[1]

    while len(selected_ids) < n_elites:

        candidates_ids = [x for x in range(len(genomes)) if x not in selected_ids]
        for selected_id in selected_ids:
            fitness_values = fitness_values.at[selected_id].set(jnp.ones(num_cases) * -jnp.inf)

        # Create a shuffled list of case indices
        case_indices = jnp.arange(num_cases)
        rnd_key, shuffle_key = random.split(rnd_key, 2)
        case_indices = random.permutation(shuffle_key, case_indices)

        # Iterate through the shuffled test cases
        for case_idx in case_indices:
            # If only one candidate remains, return it
            if len(candidates_ids) == 1:
                selected_ids.append(candidates_ids[0])
                break

            # Extract the error scores for the current case across candidates
            case_fitness_values = fitness_values[:, case_idx]

            # Filter candidates to only include those with the max reward
            candidates_ids = jnp.argwhere(case_fitness_values == jnp.max(case_fitness_values))

        selected_ids.extend(candidates_ids)

    return genomes[selected_ids[0:n_elites]]
