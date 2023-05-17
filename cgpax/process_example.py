import jax.numpy

from cgpax.jax_encoding import genome_to_program, genome_and_mask_to_program
from cgpax.individual import Individual
from functions import plus, minus, times, prot_div

cfg = {
    "n_nodes": 20,
    "functions": [plus, minus, times, prot_div],
    "constrained": False,
    "p_mut_inputs": 0.2,
    "p_mut_functions": 0.2,
    "p_mut_outputs": 0.2
}

cfg_2 = {
    "n_in" : 3,
    "n_out" : 2,
    "n_nodes": 20,
    "functions": ["plus", "minus", "times", "prot_div"],
    "constrained": False,
    "p_mut_inputs": 0.2,
    "p_mut_functions": 0.2,
    "p_mut_outputs": 0.2
}

if __name__ == '__main__':
    individual = Individual.from_config(cfg, 3, 2)
    func_name = "my_process"
    # dct = individual.exec_process_program(function_name=func_name)
    dct = {}
    program = individual.get_process_program(function_name=func_name)
    print(program)
    exec(program, dct)
    print(dct[func_name])

    buffer = jax.numpy.zeros(len(individual.buffer))
    ins = jax.numpy.ones(3)
    jit_new_buffer, jit_outputs = dct[func_name](ins, buffer)
    print(jit_outputs)
    print(jit_new_buffer)
    print()

    std_outputs = individual.process(ins)
    print(std_outputs)
    print(individual.buffer)

    genome_and_mask = individual.get_genome_and_active_mask()
    graph_program = genome_and_mask_to_program(genome_and_mask, cfg_2)
    print(graph_program)
    buffer = jax.numpy.zeros(len(individual.buffer))
    ins = jax.numpy.ones(3)
    graph_buffer, graph_outputs = graph_program(ins, buffer)
    print(graph_outputs)
    print(graph_buffer)
    print()
