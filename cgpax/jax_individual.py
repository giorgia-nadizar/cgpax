from functools import partial
from typing import Tuple, Callable

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


def compute_cgp_mutation_prob_mask(config: dict, n_out: int) -> jnp.ndarray:
    in_mut_mask = config["p_mut_inputs"] * jnp.ones(config["n_nodes"])
    f_mut_mask = config["p_mut_functions"] * jnp.ones(config["n_nodes"])
    out_mut_mask = config["p_mut_outputs"] * jnp.ones(n_out)
    return jnp.concatenate((in_mut_mask, in_mut_mask, f_mut_mask, out_mut_mask))


def compute_lgp_mutation_prob_mask(config: dict) -> jnp.ndarray:
    n_rows = config["n_rows"]
    lhs_mask = config["p_mut_lhs"] * jnp.ones(n_rows)
    rhs_mask = config["p_mut_rhs"] * jnp.ones(n_rows)
    f_mask = config["p_mut_functions"] * jnp.ones(n_rows)
    return jnp.concatenate((lhs_mask, rhs_mask, rhs_mask, f_mask))


def compute_cgp_genome_mask(config: dict, n_in: int, n_out: int) -> jnp.ndarray:
    n_nodes = config["n_nodes"]
    if config["recursive"]:
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


def compute_lgp_genome_mask(config: dict, n_in: int) -> jnp.ndarray:
    n_rows = config["n_rows"]
    n_registers = config["n_registers"]
    lhs_mask = (n_registers - n_in) * jnp.ones(n_rows)
    rhs_mask = n_registers * jnp.ones(n_rows)
    f_mask = config["n_functions"] * jnp.ones(n_rows)
    return jnp.concatenate((lhs_mask, rhs_mask, rhs_mask, f_mask))


def generate_genome(genome_mask: jnp.ndarray, rnd_key: random.PRNGKey,
                    genome_transformation_function: Callable[[jnp.ndarray], jnp.ndarray] = identity) -> jnp.ndarray:
    random_genome = random.uniform(key=rnd_key, shape=genome_mask.shape)
    integer_genome = jnp.floor(random_genome * genome_mask).astype(int)
    return genome_transformation_function(integer_genome)


def generate_population(pop_size: int, genome_mask: jnp.ndarray, rnd_key: random.PRNGKey,
                        genome_transformation_function: Callable[
                            [jnp.ndarray], jnp.ndarray] = identity) -> jnp.ndarray:
    subkeys = random.split(rnd_key, pop_size)
    partial_generate_genome = partial(generate_genome, genome_mask=genome_mask,
                                      genome_transformation_function=genome_transformation_function)
    vmap_generate_genome = vmap(partial_generate_genome)
    return vmap_generate_genome(rnd_key=subkeys)


def lgp_one_point_crossover_genomes(genome1: jnp.ndarray, genome2: jnp.ndarray,
                                    rnd_key: random.PRNGKey) -> Tuple[jnp.ndarray, jnp.ndarray]:
    assert len(genome1) == len(genome2)
    rnd_key, xover_key = random.split(rnd_key, 2)
    chunks1 = jnp.split(genome1, 4)
    chunks2 = jnp.split(genome2, 4)
    crossover_point = random.randint(xover_key, (1, 0), 0, int(len(genome1) / 4))
    before = jnp.arange(0, crossover_point)
    after = jnp.arange(crossover_point, int(len(genome1) / 4))
    offspring1 = []
    offspring2 = []
    for chunk_idx in range(len(chunks1)):
        offspring1.append(chunks1[chunk_idx].take(before))
        offspring1.append(chunks2[chunk_idx].take(after))
        offspring2.append(chunks2[chunk_idx].take(before))
        offspring2.append(chunks1[chunk_idx].take(after))
    return jnp.concatenate(offspring1), jnp.concatenate(offspring2)


def one_point_crossover_genomes(genome1: jnp.ndarray, genome2: jnp.ndarray,
                                rnd_key: random.PRNGKey) -> Tuple[jnp.ndarray, jnp.ndarray]:
    assert len(genome1) == len(genome2)
    rnd_key, xover_key = random.split(rnd_key, 2)
    crossover_point = random.randint(xover_key, (1, 0), 0, len(genome1))
    before = jnp.arange(0, crossover_point)
    after = jnp.arange(crossover_point, len(genome1))
    offspring1 = jnp.concatenate((genome1.take(before), genome2.take(after)))
    offspring2 = jnp.concatenate((genome2.take(before), genome1.take(after)))
    return offspring1, offspring2


def mutate_genome(genome: jnp.ndarray, rnd_key: random.PRNGKey, genome_mask: jnp.ndarray, mutation_mask: jnp.ndarray,
                  genome_transformation_function: Callable[[jnp.ndarray], jnp.ndarray] = identity) -> jnp.ndarray:
    prob_key, new_genome_key = random.split(rnd_key, 2)
    new_genome = generate_genome(genome_mask, new_genome_key, genome_transformation_function)
    mutation_probs = random.uniform(key=rnd_key, shape=mutation_mask.shape)
    old_ids = (mutation_probs >= mutation_mask)
    new_ids = (mutation_probs < mutation_mask)
    return jnp.floor(genome * old_ids + new_ids * new_genome).astype(int)


def mutate_genome_n_times_stacked(genome: jnp.ndarray, rnd_key: random.PRNGKey, n_mutations: int,
                                  genome_mask: jnp.ndarray, mutation_mask: jnp.ndarray,
                                  genome_transformation_function: Callable[
                                      [jnp.ndarray], jnp.ndarray] = identity) -> jnp.ndarray:
    def mutate_and_store(idx, carry):
        genomes, genome, rnd_key = carry
        rnd_key, mutation_key = random.split(rnd_key, 2)
        new_genome = mutate_genome(genome, mutation_key, genome_mask, mutation_mask, genome_transformation_function)
        genomes = genomes.at[idx].set(new_genome)
        return genomes, new_genome, rnd_key

    genomes = jnp.zeros((n_mutations, len(genome)), dtype=int)
    mutated_genomes, _, _, _, _ = fori_loop(0, n_mutations, mutate_and_store, (genomes, genome, rnd_key))
    return mutated_genomes


def mutate_genome_n_times(genome: jnp.ndarray, rnd_key: random.PRNGKey, n_mutations: int, genome_mask: jnp.ndarray,
                          mutation_mask: jnp.ndarray,
                          genome_transformation_function: Callable[
                              [jnp.ndarray], jnp.ndarray] = identity) -> jnp.ndarray:
    subkeys = random.split(rnd_key, n_mutations)
    partial_mutate_genome = partial(mutate_genome, genome=genome, genome_mask=genome_mask, mutation_mask=mutation_mask,
                                    genome_transformation_function=genome_transformation_function)
    vmap_mutate_genome = vmap(partial_mutate_genome)
    return vmap_mutate_genome(rnd_key=subkeys)
