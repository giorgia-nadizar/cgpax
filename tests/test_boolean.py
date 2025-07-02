from functools import partial

import jax.numpy as jnp

from cgpax.boolean_evaluation import evaluate_cgp_genome
from cgpax.functions import function_set_boolean
from cgpax.utils import cgp_expression_from_genome

boolean_functions_list = ["and", "or", "xor", "and_not"]

inputs = []
outputs = []
for a1 in range(2):
    for a0 in range(2):
        for b1 in range(2):
            for b0 in range(2):
                inputs.append(jnp.asarray([a1, a0, b1, b0]))
                s0 = function_set_boolean["xor"](a0, b0)
                int0 = function_set_boolean["and"](a0, b0)
                int1 = function_set_boolean["xor"](a1, b1)
                int2 = function_set_boolean["and"](a1, b1)
                s1 = function_set_boolean["xor"](int0, int1)
                int3 = function_set_boolean["and"](int0, int1)
                c = function_set_boolean["or"](int2, int3)
                outputs.append(jnp.asarray([c, s1, s0]))

print(jnp.vstack(inputs).astype(int))

variables = {
    "i0": "a1",
    "i1": "a0",
    "i2": "b1",
    "i3": "b0",
    "o0": "c",
    "o1": "s1",
    "o2": "s0",
}

nodes = list(range(4, 11))
output_nodes = [10, 8, 4]
# outputs = [6, 4, 0]
function_genes = [2, 0, 2, 0, 2, 0, 1]
x_genes = [1, 1, 0, 0, 5, 5, 7]
y_genes = [3, 3, 2, 2, 6, 6, 9]
genome = jnp.asarray(x_genes + y_genes + function_genes + output_nodes)

config = {"n_in": len(inputs[0]), "n_nodes": len(function_genes), "n_out": len(output_nodes), "n_constants": 0,
          "buffer_size": len(function_genes) + len(inputs[0])}
expr = cgp_expression_from_genome(genome, config)
for var in variables.keys():
    expr = expr.replace(var, variables[var])

print(expr, "\n")

genome_evaluation_function = evaluate_cgp_genome
genome_to_fitness = partial(genome_evaluation_function, config=config, x_values=jnp.vstack(inputs),
                            y_values=jnp.asarray(outputs))

print(genome_to_fitness(genome))
