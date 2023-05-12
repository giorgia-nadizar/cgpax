import jax.numpy
from jax import jit
from jax.lax import fori_loop
from cgpax.individual import Individual
from functions import plus, minus, times, prot_div

cfg = {
    "n_nodes": 10,
    "functions": [plus, minus, times, prot_div],
    "constrained": False,
    "p_mut_inputs": 0.2,
    "p_mut_functions": 0.2,
    "p_mut_outputs": 0.2
}

if __name__ == '__main__':
    individual = Individual.from_config(cfg, 3, 2)
    txt = individual.get_process_program("my_process")
    exec(txt)
    buffer = jax.numpy.zeros(10)
    ins = jax.numpy.ones(3)
    new_buffer, outputs = my_process(ins, buffer)
    print(outputs)
    print(individual.process(ins))
