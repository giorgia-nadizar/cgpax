from functools import partial
from typing import Tuple, Callable, Dict

from jax import vmap
import jax.numpy as jnp
from jax import random
from jax.lax import fori_loop

from cgpax.utils import identity


def levels_back_transformation_function(n_in: int, n_nodes: int) -> Callable[[jnp.ndarray], jnp.ndarray]:
    def genome_transformation_function(genome: jnp.ndarray) -> jnp.ndarray:
        x_genes, y_genes, other_genes = jnp.split(genome, [n_nodes, 2 * n_nodes])
        x_genes = jnp.arange(n_in, n_in + n_nodes) - x_genes - 1
        y_genes = jnp.arange(n_in, n_in + n_nodes) - y_genes - 1
        return jnp.concatenate((x_genes, y_genes, other_genes))

    return genome_transformation_function


def compute_cgp_mutation_prob_mask(config: Dict, n_out: int) -> jnp.ndarray:
    in_mut_mask = config["p_mut_inputs"] * jnp.ones(config["n_nodes"])
    f_mut_mask = config["p_mut_functions"] * jnp.ones(config["n_nodes"])
    out_mut_mask = config["p_mut_outputs"] * jnp.ones(n_out)
    return jnp.concatenate((in_mut_mask, in_mut_mask, f_mut_mask, out_mut_mask))


def compute_lgp_mutation_prob_mask(config: Dict) -> jnp.ndarray:
    n_rows = config["n_rows"]
    lhs_mask = config["p_mut_lhs"] * jnp.ones(n_rows)
    rhs_mask = config["p_mut_rhs"] * jnp.ones(n_rows)
    f_mask = config["p_mut_functions"] * jnp.ones(n_rows)
    return jnp.concatenate((lhs_mask, rhs_mask, rhs_mask, f_mask))


def compute_cgp_genome_mask(config: Dict, n_in: int, n_out: int) -> jnp.ndarray:
    n_nodes = config["n_nodes"]
    if config.get("recursive", False):
        in_mask = (n_in + n_nodes) * jnp.ones(n_nodes)
    elif config.get("levels_back") is not None:
        in_mask = jnp.minimum(
            config["levels_back"] * jnp.ones(n_nodes),
            jnp.arange(n_in, n_in + n_nodes)
        )
    else:
        in_mask = jnp.arange(n_in, n_in + n_nodes)
    f_mask = config["n_functions"] * jnp.ones(n_nodes)
    out_mask = (n_in + n_nodes) * jnp.ones(n_out)
    return jnp.concatenate((in_mask, in_mask, f_mask, out_mask))


def compute_lgp_genome_mask(config: Dict, n_in: int) -> jnp.ndarray:
    n_rows = config["n_rows"]
    n_registers = config["n_registers"]
    lhs_mask = (n_registers - n_in) * jnp.ones(n_rows)
    rhs_mask = n_registers * jnp.ones(n_rows)
    f_mask = config["n_functions"] * jnp.ones(n_rows)
    return jnp.concatenate((lhs_mask, rhs_mask, rhs_mask, f_mask))


def generate_genome(genome_mask: jnp.ndarray, rnd_key: random.PRNGKey,
                    weights_mutation_function: Callable[[random.PRNGKey], jnp.ndarray],
                    genome_transformation_function: Callable[[jnp.ndarray], jnp.ndarray] = identity,
                    ) -> jnp.ndarray:
    int_key, float_key = random.split(rnd_key, 2)
    random_genome = random.uniform(key=int_key, shape=genome_mask.shape)
    integer_genome = jnp.floor(random_genome * genome_mask).astype(int)
    transformed_integer_genome = genome_transformation_function(integer_genome)
    weights = weights_mutation_function(float_key)
    weights += jnp.ones_like(weights)
    return jnp.concatenate((transformed_integer_genome, weights))


def generate_population(pop_size: int, genome_mask: jnp.ndarray, rnd_key: random.PRNGKey,
                        weights_mutation_function: Callable[[random.PRNGKey], jnp.ndarray],
                        genome_transformation_function: Callable[
                            [jnp.ndarray], jnp.ndarray] = identity) -> jnp.ndarray:
    subkeys = random.split(rnd_key, pop_size)
    partial_generate_genome = partial(generate_genome, genome_mask=genome_mask,
                                      weights_mutation_function=weights_mutation_function,
                                      genome_transformation_function=genome_transformation_function)
    vmap_generate_genome = vmap(partial_generate_genome)
    return vmap_generate_genome(rnd_key=subkeys)


def lgp_one_point_crossover_genomes(genome1: jnp.ndarray, genome2: jnp.ndarray,
                                    rnd_key: random.PRNGKey) -> Tuple[jnp.ndarray, jnp.ndarray]:
    assert len(genome1) == len(genome2)
    rnd_key, xover_key, alpha_key = random.split(rnd_key, 3)
    chunk_size = int(len(genome1) / 5)
    crossover_point = random.randint(xover_key, [1], 0, chunk_size)
    ids = jnp.arange(len(genome1))
    mask1 = (ids < crossover_point) \
            | ((ids >= chunk_size) & (ids < chunk_size + crossover_point)) \
            | ((ids >= 2 * chunk_size) & (ids < 2 * chunk_size + crossover_point)) \
            | ((ids >= 3 * chunk_size) & (ids < 3 * chunk_size + crossover_point))
    mask2 = jnp.invert(mask1)
    new_lgp_genome1, _ = jnp.split(jnp.where(mask1, genome1, genome2), [4 * chunk_size])
    new_lgp_genome2, _ = jnp.split(jnp.where(mask2, genome1, genome2), [4 * chunk_size])

    alpha1, alpha2 = tuple(random.uniform(alpha_key, [2]))
    _, constants1 = jnp.split(genome1, [4 * chunk_size])
    _, constants2 = jnp.split(genome2, [4 * chunk_size])

    new_constants_1 = alpha1 * constants1 + (1 - alpha1) * constants2
    new_constants_2 = alpha2 * constants1 + (1 - alpha2) * constants2

    new_genome1 = jnp.concatenate([new_lgp_genome1, new_constants_1])
    new_genome2 = jnp.concatenate([new_lgp_genome2, new_constants_2])

    return new_genome1, new_genome2


def mutate_genome(genome: jnp.ndarray, rnd_key: random.PRNGKey, genome_mask: jnp.ndarray, mutation_mask: jnp.ndarray,
                  weights_mutation_function: Callable[[random.PRNGKey], jnp.ndarray],
                  genome_transformation_function: Callable[[jnp.ndarray], jnp.ndarray] = identity,
                  ) -> jnp.ndarray:
    old_int_genome, old_weights_genome = jnp.split(genome, [len(genome_mask)])
    prob_key, new_genome_key = random.split(rnd_key, 2)
    new_genome = generate_genome(genome_mask, new_genome_key, weights_mutation_function, genome_transformation_function)
    new_int_genome, new_weights_genome = jnp.split(new_genome, [len(genome_mask)])
    mutation_probs = random.uniform(key=rnd_key, shape=mutation_mask.shape)
    old_ids = (mutation_probs >= mutation_mask)
    new_ids = (mutation_probs < mutation_mask)
    mutated_integer_genome = jnp.floor(old_int_genome * old_ids + new_ids * new_int_genome).astype(int)
    mutated_weights = old_weights_genome + new_weights_genome
    mutated_weights -= jnp.ones_like(old_weights_genome)
    return jnp.concatenate([mutated_integer_genome, mutated_weights])


def mutate_genome_n_times_stacked(genome: jnp.ndarray, rnd_key: random.PRNGKey, n_mutations: int,
                                  genome_mask: jnp.ndarray, mutation_mask: jnp.ndarray,
                                  weights_mutation_function: Callable[[random.PRNGKey], jnp.ndarray],
                                  genome_transformation_function: Callable[
                                      [jnp.ndarray], jnp.ndarray] = identity) -> jnp.ndarray:
    def _mutate_and_store(idx, carry):
        genomes, genome, rnd_key = carry
        rnd_key, mutation_key = random.split(rnd_key, 2)
        new_genome = mutate_genome(genome, mutation_key, genome_mask, mutation_mask, weights_mutation_function,
                                   genome_transformation_function)
        genomes = genomes.at[idx].set(new_genome)
        return genomes, new_genome, rnd_key

    genomes = jnp.zeros((n_mutations, len(genome)), dtype=int)
    mutated_genomes, _, _, _, _ = fori_loop(0, n_mutations, _mutate_and_store, (genomes, genome, rnd_key))
    return mutated_genomes


def mutate_genome_n_times(genome: jnp.ndarray, rnd_key: random.PRNGKey, n_mutations: int, genome_mask: jnp.ndarray,
                          mutation_mask: jnp.ndarray,
                          weights_mutation_function: Callable[[random.PRNGKey], jnp.ndarray],
                          genome_transformation_function: Callable[
                              [jnp.ndarray], jnp.ndarray] = identity) -> jnp.ndarray:
    subkeys = random.split(rnd_key, n_mutations)
    partial_mutate_genome = partial(mutate_genome, genome=genome, genome_mask=genome_mask, mutation_mask=mutation_mask,
                                    weights_mutation_function=weights_mutation_function,
                                    genome_transformation_function=genome_transformation_function)
    vmap_mutate_genome = vmap(partial_mutate_genome)
    return vmap_mutate_genome(rnd_key=subkeys)
