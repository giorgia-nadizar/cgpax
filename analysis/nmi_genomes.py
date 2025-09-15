from functools import partial
from pathlib import Path
import jax.numpy as jnp

import yaml
from jax import vmap, random

from cgpax.gomea.fos import _compute_normalized_mutual_information_matrix
from cgpax.run_utils import compute_genome_transformation_function, compute_masks
from cgpax.standard import individual
from cgpax.utils import compute_active_genome


def column_unique_counts_vmap(matrix: jnp.ndarray):
    def count_unique(col):
        return jnp.unique(col, size=matrix.shape[0]).shape[0]

    return vmap(count_unique)(matrix.T)


if __name__ == '__main__':
    seed = 0
    environment = "inverted_double_pendulum"
    pop_type = "_large_pop"

    base_path = f"../results/gomea_cgp_{environment}_LT_{seed}_fi{pop_type}"
    config = yaml.safe_load(Path(f"{base_path}/config.yml").read_text())
    genomes = jnp.load(Path(f"{base_path}/genomes.npy"))

    # compute genomes activity histogram
    active_genome_compute_fn = partial(compute_active_genome, config=config)
    active_genomes = jnp.asarray([active_genome_compute_fn(g) for g in genomes])
    activity_count = jnp.sum(active_genomes, axis=0)
    jnp.save(f"{base_path}/activity_count.npy", activity_count)
    print("genomes activity")

    # compute unique genes
    unique_counts = column_unique_counts_vmap(genomes)
    jnp.save(Path(f"{base_path}/unique_genes.npy"), unique_counts)
    print("genomes uniqueness")

    # compute nmi matrix
    # genome_mask, mutation_mask = compute_masks(config)
    # genome_transformation_function = compute_genome_transformation_function(config)
    # rnd_key = random.PRNGKey(0)
    # init_genomes = individual.generate_population(pop_size=config["n_individuals"],
    #                                               genome_mask=genome_mask, rnd_key=rnd_key,
    #                                               genome_transformation_function=genome_transformation_function)
    # _, bias_matrix = _compute_normalized_mutual_information_matrix(init_genomes, config, None)
    bias_matrix = jnp.ones((len(genomes[0]), len(genomes[0])))
    nmi_matrix, _ = _compute_normalized_mutual_information_matrix(genomes, config, bias_matrix)
    jnp.save(f"{base_path}/nmi_matrix.npy", nmi_matrix)
    print("nmi matrix")
