from copy import deepcopy
from functools import partial

import jax.numpy as jnp
from typing import List, Tuple, Callable, Union

from jax import random, jit
import random as rnd


@jit
def _gom_mutate(genotype: jnp.ndarray, fos: jnp.ndarray, rnd_key: random.PRNGKey, donors: jnp.ndarray) -> jnp.ndarray:
    donor = random.choice(rnd_key, donors)
    return genotype.at[fos].set(donor[fos])


def parallel_gom(
        donors: jnp.ndarray,
        fitnesses: jnp.ndarray,
        fos: List[List[int]],
        eval_fn: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray],
        rnd_key: random.PRNGKey,
        track_fitnesses: bool = False,
        intermediate_prints: bool = False
) -> Union[Tuple[jnp.ndarray, jnp.ndarray], Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]]:
    mutation_fn = partial(_gom_mutate, donors=donors)
    array_fos = [jnp.asarray(f) for f in fos]
    shuffled_fos = [rnd.sample(array_fos, len(array_fos)) for _ in donors]

    genotypes = deepcopy(donors)
    fitnesses_history = []

    for f_idx in range(len(array_fos)):
        rnd_key, *mutate_keys = random.split(rnd_key, len(donors) + 1)
        offspring_genotypes = jnp.asarray([mutation_fn(genotype, shuffled_fos[g_idx][f_idx], mutate_keys[g_idx])
                                           for g_idx, genotype in enumerate(genotypes)])
        rnd_key, *eval_keys = random.split(rnd_key, len(genotypes) + 1)

        offspring_fitnesses = eval_fn(offspring_genotypes, jnp.array(eval_keys))

        genotypes = jnp.where((offspring_fitnesses > fitnesses)[:, None], offspring_genotypes, genotypes)
        fitnesses = jnp.where(offspring_fitnesses > fitnesses, offspring_fitnesses, fitnesses)

        if intermediate_prints:
            print(f"\t {f_idx} \t FITNESS: {jnp.max(fitnesses)}")
        fitnesses_history.append(jnp.max(fitnesses))

    if track_fitnesses:
        return genotypes, fitnesses, jnp.asarray(fitnesses_history)
    else:
        return genotypes, fitnesses
