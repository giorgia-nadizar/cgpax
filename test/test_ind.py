from cgpax import get_config, Individual
import pytest

cfg = get_config("cfg/test.yaml")

def test_create_ind():
    n_in = 3
    n_out = 4
    ind = Individual(cfg, n_in, n_out)
    assert ind.n_in == n_in
    assert ind.n_out == n_out
    assert len(ind.buffer) == cfg['n_nodes'] + n_in + n_out
    assert np.all(ind.buffer == 0)
    ind2 = Individual(cfg, n_in, n_out)
    assert not (ind is ind2)
    assert np.any(ind.x_genes != ind2.x_genes)
    assert np.any(ind.y_genes != ind2.y_genes)
    assert np.any(ind.f_genes != ind2.f_genes)
   
def test_call_ind():
    n_in = 1
    n_out = 2
    ind = Individual(cfg, n_in, n_out)
    inputs = np.random.rand(n_in)
    outputs = ind.process(inputs)
    assert len(outputs) == n_out
    if cfg['constrained']:
        assert np.max(outputs) <= 1.0
        assert np.min(outputs) >= -1.0
    ind.reset()
    assert np.all(ind.buffer == 0)
    # only works if test.yaml functions are add, sub, mult, div:
    if len(cfg['functions'] == 4):
        inputs = np.zeros(n_in)
        outputs = ind.process(inputs)
        assert np.all(outputs == 0)