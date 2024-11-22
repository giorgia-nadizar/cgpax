from copy import deepcopy
from functools import partial

import jax.numpy as jnp
from typing import Tuple, Dict, List

from jax import lax, vmap, random, jit


@partial(jit, static_argnames=("mpm_length",))
def _nearest_neighbor(idx: int, s_matrix: jnp.ndarray, n_indices: jnp.ndarray, mpm_length: int) -> int:
    result = jnp.where(idx == 0, 1, 0)

    def _loop_fn(res, i):
        better_candidate = (
                (s_matrix[idx, i] > s_matrix[idx, res]) |
                ((s_matrix[idx, i] == s_matrix[idx, res]) & (n_indices[i] < n_indices[res]))
        )
        res = jnp.where((i != idx) & better_candidate, i, res)
        return res, None

    result, _ = lax.scan(_loop_fn, result, jnp.arange(1, mpm_length))

    return result


# https://github.com/marcovirgolin/gpg/blob/pybind/src/fos.hpp#L284
def compute_fos(normalized_information_matrix: jnp.ndarray, rnd_key: random.PRNGKey) -> List:
    n_entries = normalized_information_matrix.shape[0]
    rnd_key, permutation_key = random.split(rnd_key)
    random_order = random.permutation(permutation_key, n_entries)

    # initial marginal product model
    mpm_length = n_entries
    mpm = [[random_order[i].item()] for i in range(n_entries)]
    mpm_n_indices = jnp.ones(n_entries, dtype=jnp.int32)

    # initialize hierarchical cluster to the initial MPM
    h_cluster = deepcopy(mpm)
    h_cluster += [[] for _ in range(n_entries - 1)]
    cl_idx = mpm_length

    # rearrange similarity matrix based on random order of MPM
    sprime = jnp.zeros(normalized_information_matrix.shape)
    for i in range(mpm_length):
        for j in range(mpm_length):
            sprime = sprime.at[i, j].set(
                normalized_information_matrix[mpm[i][0], mpm[j][0]],
            )

    nn_chain = jnp.zeros(n_entries + 2, dtype=jnp.int32)
    nn_chain_length = 0
    done = False

    while not done:

        if nn_chain_length == 0:
            rnd_key, length_key = random.split(rnd_key)
            nn_chain = nn_chain.at[nn_chain_length].set(
                random.randint(length_key, shape=(), minval=0, maxval=mpm_length))
            nn_chain_length += 1

        while nn_chain_length < 3:
            nn_chain = nn_chain.at[nn_chain_length].set(
                _nearest_neighbor(nn_chain[nn_chain_length - 1].item(), sprime, mpm_n_indices, mpm_length))
            nn_chain_length += 1

        while nn_chain[nn_chain_length - 3] != nn_chain[nn_chain_length - 1]:
            nn_chain = nn_chain.at[nn_chain_length].set(
                _nearest_neighbor(nn_chain[nn_chain_length - 1].item(), sprime, mpm_n_indices, mpm_length)
            )
            if (nn_chain[nn_chain_length].item() != nn_chain[nn_chain_length - 2].item() and
                    sprime[nn_chain[nn_chain_length - 1], nn_chain[nn_chain_length]].item() ==
                    sprime[nn_chain[nn_chain_length - 1], nn_chain[nn_chain_length - 2]].item()):
                nn_chain = nn_chain.at[nn_chain_length].set(nn_chain[nn_chain_length - 2].item())
            nn_chain_length += 1
            if nn_chain_length > n_entries:
                break

        r0 = nn_chain[nn_chain_length - 2].item()
        r1 = nn_chain[nn_chain_length - 1].item()
        if r0 > r1:
            r0, r1 = r1, r0
        nn_chain_length -= 3

        # test required for exceptional cases in which the nearest-neighbor ordering has changed within the chain
        # while merging within that chain
        if r1 < mpm_length:
            indices_array = [mpm[r0][j] for j in range(mpm_n_indices[r0])]
            indices_array += [mpm[r1][j] for j in range(mpm_n_indices[r1])]
            h_cluster[cl_idx] = indices_array
            cl_idx += 1

            mul0 = float(mpm_n_indices[r0].item()) / float(mpm_n_indices[r0].item() + mpm_n_indices[r1].item())
            mul1 = float(mpm_n_indices[r1].item()) / float(mpm_n_indices[r0].item() + mpm_n_indices[r1].item())
            for i in range(mpm_length):
                if i != r0 and i != r1:
                    sprime = sprime.at[i, r0].set(mul0 * sprime[i, r0] + mul1 * sprime[i, r1])
                    sprime = sprime.at[r0, i].set(sprime[i, r0])

            mpm_new_length = mpm_length - 1
            mpm_new = mpm[:mpm_new_length]
            mpm_new_n_indices = mpm_n_indices[:mpm_new_length]

            mpm_new[r0] = deepcopy(indices_array)
            mpm_new_n_indices = mpm_new_n_indices.at[r0].set(mpm_n_indices[r0] + mpm_n_indices[r1])

            if r1 < mpm_new_length:
                mpm_new[r1] = mpm[mpm_new_length]
                mpm_new_n_indices = mpm_new_n_indices.at[r1].set(mpm_n_indices[mpm_new_length])

                for i in range(mpm_new_length):
                    if i == r1:
                        continue
                    sprime = sprime.at[i, r1].set(sprime[i, mpm_new_length])
                    sprime = sprime.at[r1, i].set(sprime[i, r1])

            for i in range(nn_chain_length):
                if nn_chain[i].item() == mpm_new_length:
                    nn_chain = nn_chain.at[i].set(r1)
                    break

            mpm = mpm_new
            mpm_n_indices = mpm_new_n_indices
            mpm_length = mpm_new_length

            if mpm_length == 1:
                done = True

    return h_cluster


def compute_normalized_mutual_information_matrix(
        genomes: jnp.ndarray,
        config: Dict,
        bias_matrix: jnp.ndarray = None,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    if config["solver"] == "cgp":
        n_nodes = config["n_nodes"]
        n_in = config["n_in"]
        n_connection_symbols = n_in + n_nodes
        n_functions = config["n_functions"]
        n_symbols = n_connection_symbols + n_functions
        xy_genes, f_genes, out_genes = jnp.split(genomes, [2 * n_nodes, 3 * n_nodes], axis=1)
        f_genes_shifted = f_genes + n_connection_symbols
        shifted_genomes = jnp.concatenate([xy_genes, f_genes_shifted, out_genes], axis=1)
        return _compute_normalized_mutual_information_matrix(shifted_genomes, n_symbols, bias_matrix)
    else:
        raise NotImplementedError


def _compute_normalized_mutual_information_matrix(
        genomes: jnp.ndarray,
        n_symbols: int,
        bias_matrix: jnp.ndarray = None,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    pop_size, n_random_variables = genomes.shape

    def _compute_frequency_matrix(genome: jnp.ndarray, carry: [int, int, jnp.ndarray]) -> jnp.ndarray:
        i, j, frequency_matrix = carry
        frequency_matrix = frequency_matrix.at[genome.at[i].get(), genome.at[j].get()].set(1.)
        return frequency_matrix

    def _inner_nmi_fn(j: int, carry: Tuple[int, jnp.ndarray]) -> Tuple[int, jnp.ndarray]:
        i, nmi_mat = carry
        frequency_matrices = vmap(_compute_frequency_matrix, in_axes=(0, None))(genomes, (
            i, j, jnp.zeros((n_symbols, n_symbols))))
        frequency_matrix = jnp.sum(frequency_matrices, axis=0)
        flat_frequency_matrix = jnp.ravel(frequency_matrix) / pop_size
        log_flat_frequency_matrix = jnp.nan_to_num(jnp.log(flat_frequency_matrix), neginf=0.)
        entropy_flat_matrix = jnp.multiply(flat_frequency_matrix, log_flat_frequency_matrix)
        entropy = jnp.sum(entropy_flat_matrix)
        nmi_mat = nmi_mat.at[i, j].set(nmi_mat.at[i, j].get() - entropy)
        nmi_mat = nmi_mat.at[j, i].set(nmi_mat.at[i, j].get())
        return i, nmi_mat

    def _outer_nmi_fn(i: int, nmi_mat: jnp.ndarray) -> jnp.ndarray:
        _, nmi_mat = lax.fori_loop(i, n_random_variables, _inner_nmi_fn, (i, nmi_mat))
        return nmi_mat

    nmi_matrix = lax.fori_loop(0, n_random_variables, _outer_nmi_fn,
                               jnp.zeros((n_random_variables, n_random_variables), dtype=float))

    # register bias to account for non-uniform distribution of symbols in initial population
    if bias_matrix is None:
        bias_matrix = jnp.ones_like(nmi_matrix, dtype=float) * 2. - jnp.identity(n_random_variables)
        bias_matrix = jnp.divide(bias_matrix, nmi_matrix)
        bias_matrix = jnp.nan_to_num(bias_matrix, posinf=0., neginf=0., )

    # apply bias
    nmi_matrix = jnp.multiply(nmi_matrix, bias_matrix)

    # transform entropy into mutual information
    nmi_matrix_diag = nmi_matrix.diagonal()
    nmi_ii = jnp.vstack([nmi_matrix_diag] * n_random_variables)
    nmi_jj = nmi_ii.transpose()
    nmi_matrix = nmi_ii + nmi_jj - nmi_matrix

    return nmi_matrix, bias_matrix
