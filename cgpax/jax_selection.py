import jax.numpy as jnp
from jax import random


def truncation_selection(genomes: jnp.ndarray, fitness_values: jnp.ndarray, n_elites: int) -> jnp.ndarray:
    elites, _ = jnp.split(jnp.argsort(-fitness_values), [n_elites])
    return jnp.take(genomes, elites)


def fp_selection(genomes: jnp.ndarray, fitness_values: jnp.ndarray, rnd_key: random.PRNGKey,
                 n_elites: int) -> jnp.ndarray:
    p = 1 - ((jnp.max(fitness_values) - fitness_values) / (jnp.max(fitness_values) - jnp.min(fitness_values)))
    p /= jnp.sum(p)
    return random.choice(rnd_key, genomes, shape=[n_elites], p=p)
