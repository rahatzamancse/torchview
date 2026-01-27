from __future__ import annotations

from typing import Tuple, Any, Optional, TYPE_CHECKING
from collections.abc import Callable

import torch
from torch import nn
import numpy as np

from .base_node import Node, NodeContainer
from ..utils import is_generator_empty

if TYPE_CHECKING:
    from ..tensor_store import TensorStore


class TensorNode(Node):
    '''Subclass of node specialzed for nodes that
    stores tensor (subclass of torch.Tensor called RecorderTensor)
    
    Tensor data is stored to disk using memory-mapped files via TensorStore,
    avoiding memory issues with large models.
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
        tensor_store: TensorStore | None = None,
    ):

        super(TensorNode, self).__init__(
            depth, parents, children, name,
        )
        self.tensor_id = id(tensor)
        self.name = name

        # Store tensor shape (always needed)
        self._tensor_shape: tuple[int, ...] = tuple(tensor.shape)

        # Disk-based tensor storage (no memory limit)
        self.tensor_path: str | None = None
        self.tensor_dtype: str | None = None
        self._tensor_store: TensorStore | None = tensor_store
        
        if store_tensor_data and tensor_store is not None:
            # Convert tensor to regular Tensor before detaching and converting to numpy
            regular_tensor = tensor.as_subclass(torch.Tensor)
            arr = regular_tensor.detach().cpu().numpy()
            # Save to disk via memmap
            self.tensor_path = tensor_store.save(arr, str(self.tensor_id))
            self.tensor_dtype = str(arr.dtype)
        
        self.is_aux = is_aux
        self.main_node = self if main_node is None else main_node
        self.context = [] if context is None else context
        self.parent_hierarchy = {} if parent_hierarchy is None else parent_hierarchy
        self.set_node_id()
        self.collect_attributes = collect_attributes
        # Store settings for propagation through graph
        self.store_tensor_data = store_tensor_data

    def get_tensor_shape(self) -> tuple[int, ...]:
        return self._tensor_shape

    def has_tensor_data(self) -> bool:
        '''Returns True if tensor data is stored (on disk)'''
        return self.tensor_path is not None
    
    def get_tensor_data(self) -> np.ndarray | None:
        '''Load and return tensor data from disk.
        
        Returns:
            Numpy array if tensor data is stored, None otherwise.
            The returned array is memory-mapped for efficient access.
        '''
        if self.tensor_path is None or self._tensor_store is None:
            return None
        return self._tensor_store.load(self.tensor_path)
    
    def get_tensor_data_copy(self) -> np.ndarray | None:
        '''Load and return a full copy of tensor data from disk.
        
        Use this when you need to modify the data.
        
        Returns:
            Numpy array copy if tensor data is stored, None otherwise.
        '''
        if self.tensor_path is None or self._tensor_store is None:
            return None
        return self._tensor_store.load_copy(self.tensor_path)

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
