from functools import partial

from jax import vmap
import jax.numpy as jnp
from jax import random


def compute_mutation_prob_mask(config: dict, n_out: int) -> jnp.ndarray:
    in_mut_mask = config["p_mut_inputs"] * jnp.ones(config["n_nodes"])
    f_mut_mask = config["p_mut_functions"] * jnp.ones(config["n_nodes"])
    out_mut_mask = config["p_mut_outputs"] * jnp.ones(n_out)
    return jnp.concatenate((in_mut_mask, in_mut_mask, f_mut_mask, out_mut_mask))


def compute_genome_mask(config: dict, n_in: int, n_out: int) -> jnp.ndarray:
    n_nodes = config["n_nodes"]
    # TODO non-recursive only
    in_mask = jnp.arange(n_in, n_in + n_nodes)
    f_mask = len(config["functions"]) * jnp.ones(n_nodes)
    out_mask = (n_in + n_nodes) * jnp.ones(n_out)
    return jnp.concatenate((in_mask, in_mask, f_mask, out_mask))


def generate_genome(genome_mask: jnp.ndarray, rnd_key: random.PRNGKey) -> jnp.ndarray:
    random_genome = random.uniform(key=rnd_key, shape=genome_mask.shape)
    return jnp.floor(random_genome * genome_mask).astype(int)


def generate_population(pop_size: int, genome_mask: jnp.ndarray, rnd_key: random.PRNGKey) -> jnp.ndarray:
    subkeys = random.split(rnd_key, pop_size)
    partial_generate_genome = partial(generate_genome, genome_mask=genome_mask)
    vmap_generate_genome = vmap(partial_generate_genome)
    return vmap_generate_genome(rnd_key=subkeys)


def mutate_genome(genome: jnp.ndarray, rnd_key: random.PRNGKey, genome_mask: jnp.ndarray,
                  mutation_mask: jnp.ndarray) -> jnp.ndarray:
    prob_key, new_genome_key = random.split(rnd_key, 2)
    new_genome = generate_genome(genome_mask, new_genome_key)

    mutation_probs = random.uniform(key=rnd_key, shape=mutation_mask.shape)
    old_ids = jnp.where((mutation_probs >= mutation_mask), size=len(mutation_mask), fill_value=jnp.nan)[0]
    new_ids = jnp.where((mutation_probs < mutation_mask), size=len(mutation_mask), fill_value=jnp.nan)[0]
    return jnp.floor(genome * old_ids + new_ids * new_genome).astype(int)


def mutate_genome_n_times(genome: jnp.ndarray, rnd_key: random.PRNGKey, n_mutations: int, genome_mask: jnp.ndarray,
                          mutation_mask: jnp.ndarray) -> jnp.ndarray:
    subkeys = random.split(rnd_key, n_mutations)
    partial_mutate_genome = partial(mutate_genome, genome=genome, genome_mask=genome_mask, mutation_mask=mutation_mask)
    vmap_mutate_genome = vmap(partial_mutate_genome)
    return vmap_mutate_genome(rnd_key=subkeys)
