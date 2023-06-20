from typing import List, Tuple

import numpy as np
import jax.numpy as jnp
from jax import jit

from cgpax.jax_functions import available_functions

import pygraphviz as pgv


@jit
def identity(x):
    return x


def compute_active_size(genome: jnp.ndarray, config: dict) -> Tuple[int, int]:
    if config["solver"] == "cgp":
        n_nodes = config["n_nodes"]
        x_genes, y_genes, f_genes, out_genes = jnp.split(genome, jnp.asarray([n_nodes, 2 * n_nodes, 3 * n_nodes]))
        active = compute_active_graph(x_genes, y_genes, f_genes, out_genes, config)
    else:
        lhs_genes, x_genes, y_genes, f_genes = jnp.split(genome, 4)
        lhs_genes += config["n_in"]
        active = compute_coding_lines(lhs_genes, x_genes, y_genes, f_genes, config)
    return int(jnp.sum(active)), len(active)


def compute_active_graph(x_genes: jnp.ndarray, y_genes: jnp.ndarray, f_genes: jnp.ndarray, out_genes: jnp.ndarray,
                         config: dict) -> np.ndarray:
    n_nodes = config["n_nodes"]
    n_in = config["n_in"]
    active = np.zeros(n_in + n_nodes)
    active[:n_in] = True
    for i in out_genes:
        __compute_active_graph__(active, x_genes, y_genes, f_genes, n_in, i)
    return active


def __compute_active_graph__(active: np.ndarray, x_genes: jnp.ndarray, y_genes: jnp.ndarray, f_genes: jnp.ndarray,
                             n_in: int, idx: int) -> None:
    if not active[idx]:
        active[idx] = True
        __compute_active_graph__(active, x_genes, y_genes, f_genes, n_in, int(x_genes[idx - n_in]))
        arity = list(available_functions.values())[f_genes[idx - n_in]].arity
        if arity > 1:
            __compute_active_graph__(active, x_genes, y_genes, f_genes, n_in, int(y_genes[idx - n_in]))


def readable_cgp_program_from_genome(genome: jnp.ndarray, config: dict) -> str:
    n_in = config["n_in"]
    n_nodes = config["n_nodes"]
    x_genes, y_genes, f_genes, out_genes = jnp.split(genome, jnp.asarray([n_nodes, 2 * n_nodes, 3 * n_nodes]))
    active = compute_active_graph(x_genes, y_genes, f_genes, out_genes, config)
    functions = list(available_functions.values())
    text_function = f"def program(inputs, buffer):\n" \
                    f"  buffer[{list(range(n_in))}] = inputs\n"

    # execution
    for buffer_idx in range(n_in, len(active)):
        if active[buffer_idx]:
            idx = buffer_idx - n_in
            function = functions[f_genes[idx]]
            text_function += f"  buffer[{buffer_idx}] = {function.symbol}(buffer[{x_genes[idx]}]"
            if function.arity > 1:
                text_function += f", buffer[{y_genes[idx]}]"
            text_function += ")\n"

    # output selection
    text_function += f"  outputs = buffer[{out_genes}]\n"
    return text_function


def compute_coding_lines(lhs_genes: jnp.ndarray, x_genes: jnp.ndarray, y_genes: jnp.ndarray, f_genes: jnp.ndarray,
                         config: dict) -> np.ndarray:
    active = np.zeros(config["n_rows"])
    output_registers = set(range(config["n_registers"] - config["n_out"], config["n_registers"]))
    for row_id in range(config["n_rows"] - 1, -1, -1):
        if int(lhs_genes[row_id]) in output_registers:
            output_registers.remove(int(lhs_genes[row_id]))
            __compute_coding_lines__(active, lhs_genes, x_genes, y_genes, f_genes, row_id)
    return active


def __compute_coding_lines__(active: np.ndarray, lhs_genes: jnp.ndarray, x_genes: jnp.ndarray, y_genes: jnp.ndarray,
                             f_genes: jnp.ndarray, row_id: int) -> None:
    active[row_id] = True
    for new_row_id in range(row_id - 1, -1, -1):
        if lhs_genes.at[new_row_id].get() == x_genes.at[row_id].get():
            __compute_coding_lines__(active, lhs_genes, x_genes, y_genes, f_genes, new_row_id)
            break
    arity = list(available_functions.values())[f_genes.at[row_id].get()].arity
    if arity > 1:
        for new_row_id in range(row_id - 1, -1, -1):
            if lhs_genes.at[new_row_id].get() == y_genes.at[row_id].get():
                __compute_coding_lines__(active, lhs_genes, x_genes, y_genes, f_genes, new_row_id)
                break


def readable_lgp_program_from_genome(genome: jnp.ndarray, config: dict) -> str:
    lhs_genes, x_genes, y_genes, f_genes = jnp.split(genome, 4)
    lhs_genes += config["n_in"]
    functions = list(available_functions.values())
    text_function = f"def program(inputs, r):\n" \
                    f"  r[{list(range(config['n_in']))}] = inputs\n"

    active = compute_coding_lines(lhs_genes, x_genes, y_genes, f_genes, config)
    # execution
    for row_idx in range(config["n_rows"]):
        if active[row_idx]:
            function = functions[f_genes[row_idx]]
            text_function += f"  r[{lhs_genes[row_idx]}] = {function.symbol}(r[{x_genes[row_idx]}]"
            if function.arity > 1:
                text_function += f", r[{y_genes[row_idx]}]"
            text_function += ")\n"

    # output selection
    text_function += f"  outputs = r[-{config['n_out']}:]\n"
    return text_function


def __reassign_variables__(lhs_genes: jnp.ndarray, x_genes: jnp.ndarray, y_genes: jnp.ndarray, lhs_ids: List,
                           x_ids: List, y_ids: List, row_id: int, register_number: int,
                           variable_name: str) -> Tuple[List[str], List[str], List[str]]:
    current_row_id = row_id
    while current_row_id > 0:
        current_row_id -= 1
        if lhs_genes.at[current_row_id].get() == register_number:
            lhs_ids[current_row_id] = variable_name
            break
        if x_genes.at[current_row_id].get() == register_number:
            x_ids[current_row_id] = variable_name
        if y_genes.at[current_row_id].get() == register_number:
            y_ids[current_row_id] = variable_name
    return lhs_ids, x_ids, y_ids


def lgp_graph_from_genome(genome: jnp.ndarray, config: dict, x_color: str = "blue",
                          y_color: str = "orange") -> pgv.AGraph:
    lhs_genes, x_genes, y_genes, f_genes = jnp.split(genome, 4)
    lhs_genes += config['n_in']
    functions = list(available_functions.values())
    n_rows = len(lhs_genes)
    lhs_ids, x_ids, y_ids = [None] * n_rows, ["0"] * n_rows, ["0"] * n_rows

    graph = pgv.AGraph(directed=True)

    missing_outputs = set(range(config["n_registers"] - config["n_out"], config["n_registers"]))
    for row_id in range(n_rows - 1, -1, -1):
        if lhs_ids[row_id] is not None or int(lhs_genes.at[row_id].get()) in missing_outputs:
            function = functions[f_genes.at[row_id].get()]
            x_register = x_genes.at[row_id].get()
            x_name = f"g_{row_id}" if x_register > config["n_in"] else f"i_{x_register}"
            x_ids[row_id] = x_name
            lhs_ids, x_ids, y_ids = __reassign_variables__(lhs_genes, x_genes, y_genes, lhs_ids, x_ids, y_ids,
                                                           row_id, x_genes.at[row_id].get(), x_name)
            if function.arity > 1:
                y_register = y_genes.at[row_id].get()
                y_name = f"h_{row_id}" if y_register > config["n_in"] else f"i_{y_register}"
                y_ids[row_id] = y_name
                lhs_ids, x_ids, y_ids = __reassign_variables__(lhs_genes, x_genes, y_genes, lhs_ids, x_ids, y_ids,
                                                               row_id, y_genes.at[row_id].get(), y_name)
            else:
                y_ids[row_id] = None
            if int(lhs_genes.at[row_id].get()) in missing_outputs:
                missing_outputs.remove(int(lhs_genes.at[row_id].get()))
                lhs_ids[row_id] = f"{function.symbol} ({row_id})"
                graph.add_edge(f"{function.symbol} ({row_id})",
                               f'o_{config["n_out"] - (config["n_registers"] - lhs_genes.at[row_id].get())}')

    renaming_dict = {}
    for row_id in range(n_rows):
        if lhs_ids[row_id] is not None:
            function = functions[f_genes.at[row_id].get()]
            lhs = f"{function.symbol} ({row_id})"
            renaming_dict[lhs_ids[row_id]] = lhs
            x = x_ids[row_id] if x_ids[row_id].startswith("i_") else renaming_dict.get(x_ids[row_id], "0")
            graph.add_edge(x, lhs, color=x_color)
            if y_ids[row_id] is not None:
                y = y_ids[row_id] if y_ids[row_id].startswith("i_") else renaming_dict.get(y_ids[row_id], "0")
                graph.add_edge(y, lhs, color=y_color)
    return graph


def cgp_graph_from_genome(genome: jnp.ndarray, config: dict, x_color: str = "blue",
                          y_color: str = "orange") -> pgv.AGraph:
    n_in = config["n_in"]
    n_nodes = config["n_nodes"]
    x_genes, y_genes, f_genes, out_genes = jnp.split(genome, jnp.asarray([n_nodes, 2 * n_nodes, 3 * n_nodes]))
    active = compute_active_graph(x_genes, y_genes, f_genes, out_genes, config)
    functions = list(available_functions.values())

    graph = pgv.AGraph(directed=True)
    node_ids = [f"i_{n}" for n in range(n_in)]
    arities = []
    for f_id, fx in enumerate(f_genes):
        function = functions[fx]
        node_ids.append(f"{function.symbol} ({f_id + n_in})")
        arities.append(function.arity)
    for out_id, out_gene in enumerate(out_genes):
        graph.add_edge(node_ids[out_gene], f"o_{out_id}")

    for buffer_idx in range(n_in, len(active)):
        if active[buffer_idx]:
            idx = buffer_idx - n_in
            current_node = node_ids[buffer_idx]
            current_arity = arities[idx]
            graph.add_edge(node_ids[x_genes[idx]], current_node, color=x_color)
            if current_arity > 1:
                graph.add_edge(node_ids[y_genes[idx]], current_node, color=y_color)

    return graph
