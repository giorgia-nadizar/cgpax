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
        diversity_preservation: bool = False,
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
        better_offspring = offspring_fitnesses >= fitnesses
        if not diversity_preservation:
            genotypes = jnp.where(better_offspring[:, None], offspring_genotypes, genotypes)
            fitnesses = jnp.where(better_offspring, offspring_fitnesses, fitnesses)
        else:
            previous_unique_ratio = len(jnp.unique(genotypes.astype(int), axis=0)) / len(genotypes)
            diversity_preservation_mask = jnp.zeros_like(better_offspring, dtype=int)
            for g_idx in range(len(diversity_preservation_mask)):
                if not better_offspring[g_idx]:
                    continue
                tmp_div_mask = diversity_preservation_mask.at[g_idx].set(1)
                tmp_filter_mask = jnp.logical_and(better_offspring, tmp_div_mask)
                tmp_genotypes = jnp.where(tmp_filter_mask[:, None], offspring_genotypes, genotypes)
                new_unique_ratio = len(jnp.unique(tmp_genotypes.astype(int), axis=0)) / len(tmp_genotypes)
                # if the insertion of the current individual does not decrease uniqueness, proceed
                if new_unique_ratio >= previous_unique_ratio:
                    diversity_preservation_mask = tmp_div_mask
                    genotypes = tmp_genotypes
            fitnesses = jnp.where(jnp.logical_and(better_offspring, diversity_preservation_mask),
                                  offspring_fitnesses, fitnesses)

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
        diversity_preservation: bool = False,
        reference_population: jnp.ndarray = None,
        forced_improvement_ids: jnp.ndarray = None,
) -> Tuple[jnp.ndarray, jnp.ndarray, List[Dict], int]:
    if diversity_preservation:
        assert (reference_population is None) == (forced_improvement_ids is None)
    reference_population = reference_population if reference_population is not None else genotypes
    forced_improvement_ids = forced_improvement_ids if forced_improvement_ids is not None else jnp.arange(
        len(genotypes))

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

        better_offspring = offspring_fitnesses > fitnesses
        still_improving = final_fitnesses == -jnp.inf
        individuals_to_replace = jnp.logical_and(better_offspring, still_improving)
        if not diversity_preservation:
            # simulates stopping for an individual once an improvement has been found (although in practice we continue)
            final_genotypes = jnp.where(individuals_to_replace[:, None], offspring_genotypes, final_genotypes)
            final_fitnesses = jnp.where(individuals_to_replace, offspring_fitnesses, final_fitnesses)
            genotypes = jnp.where(better_offspring[:, None], offspring_genotypes, genotypes)
            fitnesses = jnp.where(better_offspring, offspring_fitnesses, fitnesses)
        else:
            diversity_preservation_mask = jnp.zeros_like(forced_improvement_ids, dtype=int)
            tmp_prev_genomes = reference_population.at[forced_improvement_ids].set(final_genotypes)
            previous_unique_ratio = len(jnp.unique(tmp_prev_genomes.astype(int), axis=0)) / len(tmp_prev_genomes)
            for g_idx in range(len(diversity_preservation_mask)):
                if not individuals_to_replace[g_idx]:
                    continue
                tmp_diversity_mask = diversity_preservation_mask.at[g_idx].set(1)
                tmp_replacement_mask = jnp.logical_and(individuals_to_replace, tmp_diversity_mask)
                tmp_final_genotypes = jnp.where(tmp_replacement_mask[:, None], offspring_genotypes, final_genotypes)
                tmp_total_pop = reference_population.at[forced_improvement_ids].set(tmp_final_genotypes)
                new_unique_ratio = len(jnp.unique(tmp_total_pop.astype(int), axis=0)) / len(tmp_total_pop)
                if new_unique_ratio >= previous_unique_ratio:
                    diversity_preservation_mask = tmp_diversity_mask
                    final_genotypes = tmp_final_genotypes
            final_fitnesses = jnp.where(jnp.logical_and(individuals_to_replace, diversity_preservation_mask),
                                        offspring_fitnesses, final_fitnesses)
            better_and_diverse = jnp.logical_and(diversity_preservation_mask, better_offspring)
            genotypes = jnp.where(better_and_diverse[:, None], offspring_genotypes, genotypes)
            fitnesses = jnp.where(better_and_diverse, offspring_fitnesses, fitnesses)

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

    if not diversity_preservation:
        # replace all individuals which did not improve with the elite individual
        final_genotypes = final_genotypes.at[jnp.where(jnp.all(final_genotypes == -jnp.inf, axis=1))[0]].set(
            elite_solution)
        final_fitnesses = jnp.where(final_fitnesses == -jnp.inf, elite_fitness, final_fitnesses)
    else:
        # keep as before
        final_genotypes = jnp.where((final_fitnesses == -jnp.inf)[:, None], genotypes, final_genotypes)
        final_fitnesses = jnp.where(final_fitnesses == -jnp.inf, fitnesses, final_fitnesses)

    assert final_genotypes.shape == genotypes.shape

    return final_genotypes, final_fitnesses, history_dicts, evaluation
