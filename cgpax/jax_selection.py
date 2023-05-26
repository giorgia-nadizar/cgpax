from functools import partial

import jax.numpy as jnp
from jax import random, vmap


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
    def single_tournament(sample_key: random.PRNGKey, genomes: jnp.ndarray, fitness_values: jnp.ndarray,
                          tour_size: int) -> jnp.ndarray:
        indexes = random.choice(sample_key, jnp.arange(start=0, stop=len(genomes)), shape=[tour_size], replace=True)
        mask = jnp.zeros_like(fitness_values)
        mask = mask.at[indexes].set(1)
        fitness_values_for_selection = (fitness_values + jnp.min(fitness_values) + 1) * mask
        best_genome = genomes.at[jnp.argmax(fitness_values_for_selection)].get()
        return best_genome

    sample_keys = random.split(rnd_key, n_elites)
    partial_single_tournament = partial(single_tournament, genomes=genomes, fitness_values=fitness_values,
                                        tour_size=tour_size)
    vmap_tournament = vmap(partial_single_tournament)
    return vmap_tournament(sample_key=sample_keys)


def island_selection(genomes: jnp.ndarray, fitness_values: jnp.ndarray, rnd_key: random.PRNGKey,
                     n_elites: int) -> jnp.ndarray:
    # parents first, then the offspring from each parent
    def select_from_ith_island(idx: int, genomes: jnp.ndarray, fitness_values: jnp.ndarray,
                               n_elites: int) -> jnp.ndarray:
        n_offspring = (len(genomes) - n_elites) / n_elites
        offspring_indexes = jnp.arange(start=0, stop=n_offspring) + n_elites + idx * n_offspring
        indexes = jnp.concatenate([jnp.array([idx]), offspring_indexes]).astype(int)
        ith_fitness = jnp.take(fitness_values, indexes)
        ith_genomes = jnp.take(genomes, indexes, axis=0)
        best_genome = ith_genomes.at[jnp.argmax(ith_fitness)].get()
        return best_genome

    partial_ith_select = partial(select_from_ith_island, genomes=genomes, fitness_values=fitness_values,
                                 n_elites=n_elites)
    vmap_ith_select = vmap(partial_ith_select)
    island_ids = jnp.arange(start=0, stop=n_elites)
    return vmap_ith_select(island_ids)
