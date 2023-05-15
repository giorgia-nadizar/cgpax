import jax.numpy
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
    buffer = jax.numpy.zeros(len(individual.buffer))
    ins = jax.numpy.ones(3)
    jit_new_buffer, jit_outputs = my_process(ins, buffer)
    print(jit_outputs)
    print(jit_new_buffer)
    print()

    std_outputs = individual.process(ins)
    print(std_outputs)
    print(individual.buffer)
