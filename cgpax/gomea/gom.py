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
        intermediate_prints: bool = False,
        test_eval_fn: Callable[[jnp.ndarray, random.PRNGKey], float] = None,
) -> Tuple[jnp.ndarray, jnp.ndarray, List[Dict]]:
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

        genotypes = jnp.where((offspring_fitnesses >= fitnesses)[:, None], offspring_genotypes, genotypes)
        fitnesses = jnp.where(offspring_fitnesses >= fitnesses, offspring_fitnesses, fitnesses)
        unique_ratio = len(jnp.unique(genotypes.astype(int), axis=0)) / len(genotypes)
        current_iteration_dict = {
            "evaluation": f_idx * len(genotypes),
            "max_fitness": jnp.max(fitnesses),
            "unique_ratio": unique_ratio
        }

        if intermediate_prints:
            print(f"\t {f_idx * len(genotypes)} \t FITNESS: {jnp.max(fitnesses)}")

        if test_eval_fn is not None:
            best_individual = genotypes[jnp.argmax(fitnesses)]
            rnd_key, test_key = random.split(rnd_key, 2)
            best_test_accuracy = test_eval_fn(best_individual, test_key)
            current_iteration_dict["test_accuracy"] = best_test_accuracy

        history_dicts.append(current_iteration_dict)

    return genotypes, fitnesses, history_dicts


@jit
def _forced_improvement_mutate(genotype: jnp.ndarray, fos: jnp.ndarray, elite_solution: jnp.ndarray) -> jnp.ndarray:
    return genotype.at[fos].set(elite_solution[fos])


def parallel_forced_improvement(
        genotypes: jnp.ndarray,
        fitnesses: jnp.ndarray,
        elite_solution: jnp.ndarray,
        elite_fitness: float,
        fos: List[List[int]],
        eval_fn: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray],
        rnd_key: random.PRNGKey,
        intermediate_prints: bool = False,
        test_eval_fn: Callable[[jnp.ndarray, random.PRNGKey], float] = None,
) -> Tuple[jnp.ndarray, jnp.ndarray, List[Dict], int]:
    print(f"Parallel Forced Improvement of {len(genotypes)} Individuals")

    forced_improvement_mutation_fn = partial(_forced_improvement_mutate, elite_solution=elite_solution)
    array_fos = [jnp.asarray(f) for f in fos]
    shuffled_fos = [rnd.sample(array_fos, len(array_fos)) for _ in genotypes]

    final_genotypes = jnp.full_like(genotypes, -jnp.inf, dtype=jnp.float32)
    final_fitnesses = jnp.full_like(fitnesses, -jnp.inf)

    evaluation = 0
    history_dicts = []

    for f_idx in range(len(array_fos)):

        rnd_key, *mutate_keys = random.split(rnd_key, len(genotypes) + 1)
        offspring_genotypes = jnp.asarray(
            [forced_improvement_mutation_fn(genotype, shuffled_fos[g_idx][f_idx])
             for g_idx, genotype in enumerate(genotypes)]
        )

        rnd_key, *eval_keys = random.split(rnd_key, len(genotypes) + 1)
        offspring_fitnesses = eval_fn(offspring_genotypes, jnp.array(eval_keys))

        currently_evaluated = jnp.sum(jnp.isneginf(final_fitnesses))

        evaluation += currently_evaluated

        # simulates stopping for an individual once an improvement has been found (although in practice we continue)
        final_genotypes = jnp.where(((offspring_fitnesses > fitnesses)[:, None]) & (final_genotypes == -jnp.inf),
                                    offspring_genotypes, final_genotypes)
        final_fitnesses = jnp.where((offspring_fitnesses > fitnesses) & (final_fitnesses == -jnp.inf),
                                    offspring_fitnesses, final_fitnesses)

        genotypes = jnp.where((offspring_fitnesses > fitnesses)[:, None], offspring_genotypes, genotypes)
        fitnesses = jnp.where(offspring_fitnesses > fitnesses, offspring_fitnesses, fitnesses)
        unique_ratio = len(jnp.unique(genotypes.astype(int), axis=0)) / len(genotypes)

        current_iteration_dict = {
            "evaluation": evaluation,
            "max_fitness": jnp.max(final_fitnesses),
            "unique_ratio": unique_ratio
        }

        if intermediate_prints:
            print(f"\t {evaluation} \t FITNESS: {jnp.max(final_fitnesses)} \t (forced improvement)")

        if test_eval_fn is not None:
            # some current individuals might be after the stopping has passed
            best_individual = final_genotypes[jnp.argmax(final_fitnesses)]
            rnd_key, test_key = random.split(rnd_key, 2)
            best_test_accuracy = test_eval_fn(best_individual, test_key)
            current_iteration_dict["test_accuracy"] = best_test_accuracy

        history_dicts.append(current_iteration_dict)

        # forced improvement has finished for all
        if jnp.all(final_fitnesses > -jnp.inf):
            break

    # replace all individuals which did not improve with the elite individual

    final_genotypes = final_genotypes.at[jnp.where(jnp.all(final_genotypes == -jnp.inf, axis=1))[0]].set(elite_solution)
    final_fitnesses = jnp.where(final_fitnesses == -jnp.inf, elite_fitness, final_fitnesses)

    return final_genotypes, final_fitnesses, history_dicts, evaluation
