import copy

from cgpax import get_config
import pytest
import numpy as np

from cgpax.individual import Individual

# cfg = get_config("cfg/test.yaml")
cfg = {
    "n_nodes": 10,
    "functions": [lambda x, y: x + y, lambda x, y: x * y, lambda x, y: x - y, lambda x, y: 0 if y == 0 else x / y],
    "constrained": False,
    "p_mut_inputs": 0.2,
    "p_mut_functions": 0.2,
    "p_mut_outputs": 0.2
}


def test_create_ind():
    n_in = 3
    n_out = 4
    ind = Individual.from_config(cfg, n_in, n_out)
    assert ind.n_in == n_in
    assert ind.n_out == n_out
    assert len(ind.buffer) == cfg['n_nodes'] + n_in
    assert np.all(ind.buffer == 0)
    ind2 = Individual.from_config(cfg, n_in, n_out)
    assert not (ind is ind2)
    # check x, y, and f genes sizes
    assert np.any(ind.x_genes != ind2.x_genes)
    assert np.any(ind.y_genes != ind2.y_genes)
    assert np.any(ind.f_genes != ind2.f_genes)


def test_call_ind():
    n_in = 1
    n_out = 2
    ind = Individual.from_config(cfg, n_in, n_out)
    inputs = np.random.rand(n_in)
    outputs = ind.process(inputs)
    assert len(outputs) == n_out
    assert np.all(ind.buffer[np.invert(ind.active)] == 0)
    if cfg['constrained']:
        assert np.max(outputs) <= 1.0
        assert np.min(outputs) >= -1.0
    ind.reset()
    assert np.all(ind.buffer == 0)
    # only works if test.yaml functions are add, sub, mult, div:
    if len(cfg['functions']) == 4:
        inputs = np.zeros(n_in)
        outputs = ind.process(inputs)
        assert np.all(outputs == 0)


def test_mutation():
    n_in = 3
    n_out = 4
    original_ind = Individual.from_config(cfg, n_in, n_out)
    copied_original_ind = copy.deepcopy(original_ind)
    mutated_ind = original_ind.mutate_from_conf(cfg)
    assert len(original_ind.buffer) == len(mutated_ind.buffer)
    assert mutated_ind is not original_ind
    assert np.all(original_ind.x_genes == copied_original_ind.x_genes)
    assert np.all(original_ind.y_genes == copied_original_ind.y_genes)
    assert np.all(original_ind.f_genes == copied_original_ind.f_genes)
    assert np.all(original_ind.out_genes == copied_original_ind.out_genes)
    inputs = np.random.rand(n_in)
    outputs = mutated_ind.process(inputs)
    assert len(outputs) == mutated_ind.n_out
    # 1 0 0 mut prob
    mutated_ind = original_ind.mutate(1, 0, 0)
    assert np.any(original_ind.x_genes != mutated_ind.x_genes)
    assert np.any(original_ind.y_genes != mutated_ind.y_genes)
    assert np.all(original_ind.f_genes == mutated_ind.f_genes)
    assert np.all(original_ind.out_genes == mutated_ind.out_genes)
    # 0 1 0 mut prob
    mutated_ind = original_ind.mutate(0, 1, 0)
    assert np.all(original_ind.x_genes == mutated_ind.x_genes)
    assert np.all(original_ind.y_genes == mutated_ind.y_genes)
    assert np.any(original_ind.f_genes != mutated_ind.f_genes)
    assert np.all(original_ind.out_genes == mutated_ind.out_genes)
    # 0 0 1 mut prob
    mutated_ind = original_ind.mutate(0, 0, 1)
    assert np.all(original_ind.x_genes == mutated_ind.x_genes)
    assert np.all(original_ind.y_genes == mutated_ind.y_genes)
    assert np.all(original_ind.f_genes == mutated_ind.f_genes)
    assert np.any(original_ind.out_genes != mutated_ind.out_genes)
