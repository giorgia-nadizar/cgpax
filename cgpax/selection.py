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
