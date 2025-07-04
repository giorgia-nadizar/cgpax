from copy import deepcopy
from functools import partial

import jax.numpy as jnp
from typing import List, Tuple, Callable, Union, Dict

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
        intermediate_prints: bool = False,
        test_eval_fn: Callable[[jnp.ndarray, random.PRNGKey], float] = None,
) -> Union[Tuple[jnp.ndarray, jnp.ndarray], Tuple[jnp.ndarray, jnp.ndarray, List[Dict]]]:
    mutation_fn = partial(_gom_mutate, donors=donors)
    array_fos = [jnp.asarray(f) for f in fos]
    shuffled_fos = [rnd.sample(array_fos, len(array_fos)) for _ in donors]

    genotypes = deepcopy(donors)
    history_dicts = []

    for f_idx in range(len(array_fos)):

        rnd_key, *mutate_keys = random.split(rnd_key, len(donors) + 1)
        offspring_genotypes = jnp.asarray([mutation_fn(genotype, shuffled_fos[g_idx][f_idx], mutate_keys[g_idx])
                                           for g_idx, genotype in enumerate(genotypes)])
        rnd_key, *eval_keys = random.split(rnd_key, len(genotypes) + 1)

        offspring_fitnesses = eval_fn(offspring_genotypes, jnp.array(eval_keys))

        genotypes = jnp.where((offspring_fitnesses > fitnesses)[:, None], offspring_genotypes, genotypes)
        fitnesses = jnp.where(offspring_fitnesses > fitnesses, offspring_fitnesses, fitnesses)
        current_iteration_dict = {
            "evaluation": f_idx * len(genotypes),
            "max_fitness": jnp.max(fitnesses),
        }

        if intermediate_prints:
            print(f"\t {f_idx * len(genotypes)} \t FITNESS: {jnp.max(fitnesses)}")

        if test_eval_fn is not None:
            best_individual = genotypes[jnp.argmax(fitnesses)]
            rnd_key, test_key = random.split(rnd_key, 2)
            best_test_accuracy = test_eval_fn(best_individual, test_key)
            current_iteration_dict["test_accuracy"] = best_test_accuracy

        history_dicts.append(current_iteration_dict)

    if track_fitnesses:
        return genotypes, fitnesses, history_dicts
    else:
        return genotypes, fitnesses
