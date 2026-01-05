from __future__ import annotations

from typing import Tuple, Any, Optional
from collections.abc import Callable

import torch
from torch import nn
import numpy as np
import warnings

from .base_node import Node, NodeContainer
from ..utils import is_generator_empty

# Default maximum tensor size in bytes (100 MB)
DEFAULT_MAX_TENSOR_BYTES = 100 * 1024 * 1024


class TensorNode(Node):
    '''Subclass of node specialzed for nodes that
    stores tensor (subclass of torch.Tensor called RecorderTensor)
    '''
    def __init__(
        self,
        tensor: torch.Tensor,
        depth: int,
        parents: NodeContainer[Node] | Node | None = None,
        children: NodeContainer[Node] | Node | None = None,
        name: str = 'tensor',
        context: Any | None = None,
        is_aux: bool = False,
        main_node: TensorNode | None = None,
        parent_hierarchy: dict[int, ModuleNode | FunctionNode] | None = None,
        collect_attributes: bool = False,
        store_tensor_data: bool = False,
        max_tensor_bytes: int = DEFAULT_MAX_TENSOR_BYTES,
    ):

        super(TensorNode, self).__init__(
            depth, parents, children, name,
        )
        self.tensor_id = id(tensor)
        self.name = name

        # Store tensor shape (always needed)
        self._tensor_shape: tuple[int, ...] = tuple(tensor.shape)

        # Optionally store full tensor data with memory limit
        self.tensor_data: np.ndarray | None = None
        if store_tensor_data:
            tensor_bytes = tensor.numel() * tensor.element_size()
            if tensor_bytes <= max_tensor_bytes:
                # Convert tensor to regular Tensor before detaching and converting to numpy
                regular_tensor = tensor.as_subclass(torch.Tensor)
                self.tensor_data = regular_tensor.detach().cpu().numpy()
            else:
                warnings.warn(
                    f"Tensor data for node '{self.name}' (id={self.tensor_id}, shape={self._tensor_shape}) "
                    f"exceeds the memory limit of {max_tensor_bytes} bytes. "
                    f"Tensor data will not be stored.",
                    RuntimeWarning
                )
        
        self.is_aux = is_aux
        self.main_node = self if main_node is None else main_node
        self.context = [] if context is None else context
        self.parent_hierarchy = {} if parent_hierarchy is None else parent_hierarchy
        self.set_node_id()
        self.collect_attributes = collect_attributes
        # Store settings for propagation through graph
        self.store_tensor_data = store_tensor_data
        self.max_tensor_bytes = max_tensor_bytes

    def get_tensor_shape(self) -> tuple[int, ...]:
        return self._tensor_shape

    def has_tensor_data(self) -> bool:
        '''Returns True if full tensor data is stored'''
        return self.tensor_data is not None

    def set_node_id(self, children_id: int | str | None = None) -> None:
        if children_id is None:
            self.node_id = (
                f'{id(self.main_node)}' if self.is_aux and self.main_node
                else f'{id(self)}'
            )
        else:
            self.node_id = f'{id(self)}-{children_id}'

class ModuleNode(Node):
    '''Subclass of node specialzed for storing torch Module info
    '''
    def __init__(
        self,
        module_unit: nn.Module,
        depth: int,
        module_name_map: dict[int, str] | None = None,
        parents: NodeContainer[Node] | Node | None = None,
        children: NodeContainer[Node] | Node | None = None,
        output_nodes: NodeContainer[Node] | None = None,
        attributes: Optional[str] = None,
    ) -> None:
        super(ModuleNode, self).__init__(
            depth, parents, children, name='module-node'
        )
        # Use precomputed module name map for O(1) lookup instead of O(n) iteration
        self.type_name = module_unit.__class__.__name__
        if module_name_map is not None:
            self.name = module_name_map.get(id(module_unit), self.type_name)
        else:
            self.name = self.type_name
        self.compute_unit_id = id(module_unit)
        self.is_activation = is_generator_empty(module_unit.parameters())
        self.is_container = not any(module_unit.children())
        self.input_shape: list[Tuple[int, ...]] = []
        self.output_shape: list[Tuple[int, ...]] = []
        self.output_nodes = NodeContainer() if output_nodes is None else output_nodes
        self.set_node_id()
        self.attributes = attributes

    def set_input_shape(self, input_shape: list[Tuple[int, ...]]) -> None:
        self.input_shape = input_shape

    def set_output_shape(self, output_shape: list[Tuple[int, ...]]) -> None:
        self.output_shape = output_shape

    def add_output_nodes(self, output_node: Node) -> None:
        self.output_nodes.add(output_node)

    def set_node_id(self, output_id: int | str | None = None) -> None:
        '''Sets the id of ModuleNode.
        If no output is given, it sets to value unique to node.
        If output id is given, there are 2 cases:
            1. Parameterless module: id is determined by output_id and id of nn.Module
            2. Module with parameter: id is determined by only id of nn.module object
        This is crucial when rolling recursive modules by identifying them with this id
        mechanism'''
        if output_id is None:
            self.node_id = f'{id(self)}'
        else:
            if self.is_activation:
                # zero-parameter module -> module for activation function, e.g. ReLU
                self.node_id = f'{self.compute_unit_id}-{output_id}'
            else:
                self.node_id = f'{self.compute_unit_id}-'


class FunctionNode(Node):
    '''Subclass of node specialized for nodes
    that does computation (e.g. torch.functions)
    '''
    def __init__(
        self,
        function_unit: Callable[..., Any],
        depth: int,
        parents: NodeContainer[Node] | Node | None = None,
        children: NodeContainer[Node] | Node | None = None,
        name: str = 'function-node',
        attributes: Optional[str] = None,
    ) -> None:
        super(FunctionNode, self).__init__(
            depth, parents, children, name
        )
        self.compute_unit_id = id(function_unit)
        self.is_container = True
        self.input_shape: list[Tuple[int, ...]] = []
        self.output_shape: list[Tuple[int, ...]] = []
        self.set_node_id()
        self.output_nodes = self.children
        self.attributes = attributes

    def set_input_shape(self, input_shape: list[Tuple[int, ...]]) -> None:
        self.input_shape = input_shape

    def set_output_shape(self, output_shape: list[Tuple[int, ...]]) -> None:
        self.output_shape = output_shape

    def add_output_nodes(self, output_node: Node) -> None:
        self.output_nodes.add(output_node)

    def set_node_id(self, output_id: int | str | None = None) -> None:
        '''Sets the id of FunctionNode.
        If no output is given, it sets to value unique to node.
        If output id is given, id is determined by only id of nn.module object
        This is crucial when rolling recursive modules by identifying them with this id
        mechanism'''
        if output_id is None:
            self.node_id = f'{id(self)}'
        else:
            self.node_id = f'{self.compute_unit_id}-{output_id}'
