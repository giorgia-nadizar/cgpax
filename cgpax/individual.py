from __future__ import annotations

import copy
from typing import List
import numpy as np


# TODO add a comment to remind that the first n_in positions in the buffer are for the inputs (and genes are shifted)
class Individual:
    n_in: int
    n_out: int
    n_nodes: int
    functions: List
    x_genes: np.ndarray
    y_genes: np.ndarray
    f_genes: np.ndarray
    out_genes: np.ndarray
    buffer: np.ndarray
    active: np.ndarray
    # TODO optimize the storage of this
    __inputs_mask__: np.ndarray

    def __init__(self, n_in: int, n_out: int, n_nodes: int, functions: List, x_genes: np.ndarray, y_genes: np.ndarray,
                 f_genes: np.ndarray, out_genes: np.ndarray, buffer: np.ndarray, inputs_mask: np.ndarray) -> None:
        self.n_in = n_in
        self.n_out = n_out
        self.n_nodes = n_nodes
        self.functions = functions
        self.x_genes = x_genes
        self.y_genes = y_genes
        self.f_genes = f_genes
        self.out_genes = out_genes
        self.buffer = buffer
        self.__inputs_mask__ = inputs_mask
        self.active = np.zeros(len(buffer), dtype=bool)
        self.compute_active()
        super().__init__()

    @classmethod
    def from_config(cls, cfg: dict, n_in: int, n_out: int):
        n_nodes = cfg["n_nodes"]
        buffer = np.zeros(n_in + n_nodes)
        # TODO this implementation is non recursive
        inputs_mask = np.arange(n_in, n_in + n_nodes)
        float_x_genes = np.random.rand(n_nodes)
        float_y_genes = np.random.rand(n_nodes)
        float_f_genes = np.random.rand(n_nodes)
        x_genes = np.floor(np.multiply(float_x_genes, inputs_mask)).astype(int)
        y_genes = np.floor(np.multiply(float_y_genes, inputs_mask)).astype(int)
        f_genes = np.floor(len(cfg["functions"]) * float_f_genes).astype(int)
        out_genes = np.floor(np.random.rand(n_out) * (n_in + n_nodes)).astype(int)
        return cls(n_in, n_out, n_nodes, cfg["functions"], x_genes, y_genes, f_genes, out_genes, buffer, inputs_mask)

    def compute_active(self):
        self.active[:self.n_in] = True
        for i in self.out_genes:
            self.__compute_active__(i)

    def __compute_active__(self, idx):
        if not self.active[idx]:
            self.active[idx] = True
            self.__compute_active__(self.x_genes[idx - self.n_in])
            self.__compute_active__(self.y_genes[idx - self.n_in])

    def reset(self) -> None:
        self.buffer[:] = 0

    def process(self, inputs: np.ndarray) -> np.ndarray:
        self.buffer[0:self.n_in] = inputs
        for buffer_idx in range(self.n_in, len(self.buffer)):
            if self.active[buffer_idx]:
                idx = buffer_idx - self.n_in
                func = self.functions[self.f_genes[idx]]
                x = self.buffer[self.x_genes[idx]]
                y = self.buffer[self.y_genes[idx]]
                self.buffer[buffer_idx] = func(x, y)
        return self.buffer[self.out_genes]

    def get_process_program(self, function_name: str = "process") -> str:
        text_function = f"""from jax import jit
global {function_name}
@jax.jit
def {function_name}(inputs, buffer):
  def copy_inputs(idx, carry):
    inputs, buffer = carry
    buffer = buffer.at[idx].set(inputs.at[idx].get())
    return inputs, buffer
  _, buffer = jax.lax.fori_loop(0, len(inputs), copy_inputs, (inputs, buffer))
"""
        function_names = [x.__name__ for x in self.functions]
        for buffer_idx in range(self.n_in, len(self.buffer)):
            if self.active[buffer_idx]:
                idx = buffer_idx - self.n_in
                text_function += f"  buffer = buffer.at[{buffer_idx}].set({function_names[self.f_genes[idx]]}(" \
                                 f"buffer.at[{self.x_genes[idx]}].get(), buffer.at[{self.y_genes[idx]}].get()))\n"

        text_function += f"  outputs = jax.numpy.zeros({self.n_out})\n"
        for outputs_idx, buffer_idx in enumerate(self.out_genes):
            text_function += f"  outputs = outputs.at[{outputs_idx}].set(buffer.at[{buffer_idx}].get())\n"

        text_function += "  return buffer, outputs"
        return text_function

    def mutate_from_config(self, cfg) -> Individual:
        return self.mutate(cfg["p_mut_inputs"], cfg["p_mut_functions"], cfg["p_mut_outputs"])

    def mutate(self, p_mut_inputs: float, p_mut_functions: float, p_mut_outputs: float) -> Individual:
        new_ind = copy.deepcopy(self)
        new_ind.reset()
        for i in range(len(self.x_genes)):
            p_x, p_y, p_f = tuple(np.random.rand(3))
            if p_x < p_mut_inputs:
                new_ind.x_genes[i] = np.random.randint(low=0, high=self.__inputs_mask__[i])
            if p_y < p_mut_inputs:
                new_ind.y_genes[i] = np.random.randint(low=0, high=self.__inputs_mask__[i])
            if p_f < p_mut_functions:
                new_ind.f_genes[i] = np.random.randint(low=0, high=len(self.functions))
        for i in range(len(self.out_genes)):
            if np.random.rand() < p_mut_outputs:
                new_ind.out_genes[i] = np.random.randint(low=0, high=len(self.buffer))
        new_ind.compute_active()
        return new_ind
