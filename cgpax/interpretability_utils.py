import re
from typing import List

from cgpax.jax_functions import available_functions

interpretability_weights = {
    "offset": 79.1,
    "l": -0.2,
    "n_o": -0.5,
    "n_nao": -3.4,
    "n_naoc": -4.5
}
arithmetic_symbols = ["+", "-", "*", "/"]


def size(formula: str) -> int:
    return count_operations(formula) + count_variables(formula) + count_constants(formula)


def count_constants(formula: str) -> int:
    return len(re.findall(r'(?<![ix])(?<!x_)\d+', formula))


def count_operations(formula: str) -> int:
    operations = 0
    for symbol in [x.symbol for x in available_functions.values()]:
        operations += formula.count(symbol)
    return operations


def count_variables(formula: str) -> int:
    return formula.count("i") + formula.count("x")


def count_non_arithmetic_operations(formula: str) -> int:
    operations = 0
    non_arithmetic_symbols = [x.symbol for x in available_functions.values() if x.symbol not in arithmetic_symbols]
    for symbol in non_arithmetic_symbols:
        operations += formula.count(symbol)
    return operations


def _rec_conn_count(connection_lists: List[List[int]], ids_lists: List[int], target: int, count: int = 1) -> int:
    for idx in range(len(connection_lists) - 1, -1, -1):
        if target in connection_lists[idx]:
            return _rec_conn_count(connection_lists[0:idx], ids_lists, ids_lists[idx], count + 1)
    return count


def count_max_chained_non_arithmetic_operations(formula: str) -> int:
    non_arithmetical_positions = []
    non_arithmetic_symbols = [x.symbol for x in available_functions.values() if x.symbol not in arithmetic_symbols]
    for symbol in non_arithmetic_symbols:
        non_arithmetical_positions.extend([m.start() for m in re.finditer(symbol, formula)])
    if len(non_arithmetical_positions) == 0:
        return 0
    non_arithmetical_positions.sort()
    connection_lists = []
    for position in non_arithmetical_positions:
        children = []
        parentheses_count = 0
        for cursor in range(position + 1, len(formula)):
            if formula[cursor] == "(":
                parentheses_count += 1
            elif formula[cursor] == ")":
                parentheses_count -= 1
            elif cursor in non_arithmetical_positions and parentheses_count > 0:
                children.append(cursor)
        connection_lists.append(children)
    depths = [_rec_conn_count(connection_lists[0:idx], non_arithmetical_positions, non_arithmetical_positions[idx])
              for idx in range(len(non_arithmetical_positions) - 1, -1, -1)]
    return max(depths)


def evaluate_interpretability(formula: str) -> float:
    phi = interpretability_weights["offset"]
    phi += interpretability_weights["l"] * size(formula)
    phi += interpretability_weights["n_o"] * count_operations(formula)
    phi += interpretability_weights["n_nao"] * count_non_arithmetic_operations(formula)
    phi += interpretability_weights["n_naoc"] * count_max_chained_non_arithmetic_operations(formula)
    return phi
