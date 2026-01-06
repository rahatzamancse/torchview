# mypy: ignore-errors
from __future__ import annotations

from collections import Counter
from contextlib import nullcontext
from typing import Any, Callable, Union

import torch.nn as nn
from graphviz import Digraph
from torch.nn.modules import Identity

from .computation_node import FunctionNode, ModuleNode, NodeContainer, TensorNode
from .computation_node.compute_node import DEFAULT_MAX_TENSOR_BYTES
from .utils import assert_input_type, updated_dict

import networkx as nx

COMPUTATION_NODES = Union[TensorNode, ModuleNode, FunctionNode]

node2color = {
    TensorNode: "lightyellow",
    ModuleNode: "darkseagreen1",
    FunctionNode: "aliceblue",
}

# TODO: Currently, we only use directed graphviz graph since DNN are
# graphs except for e.g. graph neural network (GNN). Experiment on GNN
# and see if undirected graphviz graph can be used to represent GNNs

# TODO: change api of to include also function calls, not only pytorch models
# so, keep the api here as general as possible


def build_module_name_map(model: nn.Module) -> dict[int, str]:
    '''Build a mapping from module id to its full path name.
    This is O(n) once, enabling O(1) lookups for each module.
    '''
    return {id(mod): name for name, mod in model.named_modules()}


class ComputationGraph:
    '''A class to represent Computational graph and visualization of pytorch model

    Attributes:
        visual_graph (Digraph):
            Graphviz.Digraph object to represent computational graph of
            pytorch model

        root_container (NodeContainer):
            Iterable of TensorNodes to represent all input/root nodes
            of pytorch model.

        show_shapes (bool):
            Whether to show shapes of tensor/input/outputs

        hide_module_functions (bool):
            Some modules contain only torch.function and no submodule,
            e.g. nn.Conv2d. They are usually implemented to do one type
            of computation, e.g. Conv2d -> 2D Convolution. If True,
            visual graph only displays the module itself,
            while ignoring its inner functions.

        hide_inner_tensors (bool):
            Whether to hide inner tensors in graphviz graph object


        node_hierarchy dict:
            Represents nested hierarchy of ComputationNodes by nested dictionary
    '''
    def __init__(
        self,
        visual_graph: Digraph,
        root_container: NodeContainer[TensorNode],
        show_shapes: bool = True,
        expand_nested: bool = False,
        hide_inner_tensors: bool = True,
        hide_module_functions: bool = True,
        roll: bool = True,
        depth: int | float = 3,
        collect_attributes: bool = False,
        model: nn.Module | None = None,
        store_tensor_data: bool = False,
        max_tensor_bytes: int = DEFAULT_MAX_TENSOR_BYTES,
    ):
        '''
        Resets the running_node_id, id_dict when a new ComputationGraph is initialized.
        Otherwise, labels would depend on previous ComputationGraph runs
        '''
        # Build module name map once for O(1) lookups
        if model:
            self.module_name_map: dict[int, str] = build_module_name_map(model)
        else:
            self.module_name_map = {}

        # Tensor storage settings
        self.store_tensor_data = store_tensor_data
        self.max_tensor_bytes = max_tensor_bytes

        self.visual_graph = visual_graph
        self.root_container = root_container
        self.show_shapes = show_shapes
        self.expand_nested = expand_nested
        self.hide_inner_tensors = hide_inner_tensors
        self.hide_module_functions = hide_module_functions
        self.roll = roll
        self.depth = depth
        self.collect_attributes = collect_attributes

        # specs for html table, needed for node labels
        self.html_config = {
            'border': 0,
            'cell_border': 1,
            'cell_spacing': 0,
            'cell_padding': 4,
            'col_span': 2,
            'row_span': 2,
        }
        self.reset_graph_history()

    def reset_graph_history(self) -> None:
        '''Resets to id config to the setting of empty visual graph
        needed for getting reproducible/deterministic node name and
        graphviz graphs. This is especially important for output tests
        '''
        self.context_tracker = {
            'current_context': [],
            'current_depth': 0,
            'collect_attributes': self.collect_attributes,
            'store_tensor_data': self.store_tensor_data,
            'max_tensor_bytes': self.max_tensor_bytes,
        }
        self.running_node_id: int = 0
        self.running_subgraph_id: int = 0
        self.id_dict: dict[str, int] = {}
        self.node_set: set[int] = set()
        self.edge_list: list[tuple[COMPUTATION_NODES, COMPUTATION_NODES]] = []

        # module node to capture whole graph
        main_container_module = ModuleNode(Identity(), -1)
        main_container_module.is_container = False
        self.subgraph_dict: dict[str, int] = {main_container_module.node_id: 0}
        self.running_subgraph_id += 1

        # Add input nodes
        self.node_hierarchy = {
            main_container_module: list(self.root_container)
        }
        for root_node in self.root_container:
            root_node.context = self.node_hierarchy[main_container_module]

    def fill_visual_graph(self) -> None:
        '''Fills the graphviz graph with desired nodes and edges.'''

        self.render_nodes()
        self.render_edges()
        self.resize_graph()

    def render_nodes(self) -> None:
        kwargs = {
            'cur_node': self.node_hierarchy,
            'subgraph': None,
        }
        self.traverse_graph(self.collect_graph, **kwargs)

    def render_edges(self) -> None:
        '''Records all edges in self.edge_list to
        the graphviz graph using node ids from edge_list'''
        edge_counter: dict[tuple[int, int], int] = {}
        for tail, head in self.edge_list:
            edge_id = self.id_dict[tail.node_id], self.id_dict[head.node_id]
            edge_counter[edge_id] = edge_counter.get(edge_id, 0) + 1
            self.add_edge(edge_id, edge_counter[edge_id])

    def traverse_graph(
        self, action_fn: Callable[..., None], **kwargs: Any
    ) -> None:
        cur_node = kwargs['cur_node']
        cur_subgraph = (
            self.visual_graph if kwargs['subgraph'] is None else kwargs['subgraph']
        )
        assert_input_type(
            'traverse_graph', (TensorNode, ModuleNode, FunctionNode, dict), cur_node
        )
        if isinstance(cur_node, (TensorNode, ModuleNode, FunctionNode)):
            if cur_node.depth <= self.depth:
                action_fn(**kwargs)
            return

        if isinstance(cur_node, dict):
            k, v = list(cur_node.items())[0]
            new_kwargs = updated_dict(kwargs, 'cur_node', k)
            if k.depth <= self.depth and k.depth >= 0:
                action_fn(**new_kwargs)

            # if it is container module, move directly to outputs
            if self.hide_module_functions and k.is_container:
                for g in k.output_nodes:
                    new_kwargs = updated_dict(new_kwargs, 'cur_node', g)
                    self.traverse_graph(action_fn, **new_kwargs)
                return

            display_nested = (
                k.depth < self.depth and k.depth >= 1 and self.expand_nested
            )

            with (
                cur_subgraph.subgraph(name=f'cluster_{self.subgraph_dict[k.node_id]}')
                if display_nested else nullcontext()
            ) as cur_cont:
                if display_nested:
                    cur_cont.attr(
                        style='dashed', label=f"{k.type_name}: {k.name}", labeljust='l', fontsize='12'
                    )
                    new_kwargs = updated_dict(new_kwargs, 'subgraph', cur_cont)
                for g in v:
                    new_kwargs = updated_dict(new_kwargs, 'cur_node', g)
                    self.traverse_graph(action_fn, **new_kwargs)

    def collect_graph(self, **kwargs: Any) -> None:
        '''Adds edges and nodes with appropriate node name/id (so it respects
        properties e.g. if rolled recursive nodes are given the same node name
        in graphviz graph)'''

        cur_node = kwargs['cur_node']
        # if tensor node is traced, dont repeat collecting
        # if node is isolated, dont record it
        is_isolated = cur_node.is_root() and cur_node.is_leaf()
        if id(cur_node) in self.node_set or is_isolated:
            return

        self.check_node(cur_node)
        is_cur_visible = self.is_node_visible(cur_node)
        # add node
        if is_cur_visible:
            subgraph = kwargs['subgraph']
            if isinstance(cur_node, (FunctionNode, ModuleNode)):
                if self.roll:
                    self.rollify(cur_node)
                self.add_node(cur_node, subgraph)

            if isinstance(cur_node, TensorNode):
                self.add_node(cur_node, subgraph)

        elif isinstance(cur_node, ModuleNode):
            # add subgraph
            if self.roll:
                self.rollify(cur_node)
            if cur_node.node_id not in self.subgraph_dict:
                self.subgraph_dict[cur_node.node_id] = self.running_subgraph_id
                self.running_subgraph_id += 1

        # add edges only through
        # node -> TensorNode -> Node connection
        if not isinstance(cur_node, TensorNode):
            return

        # add edges
        # {cur_node -> head} part
        tail_node = self.get_tail_node(cur_node)
        is_main_node_visible = self.is_node_visible(cur_node.main_node)
        is_tail_node_visible = self.is_node_visible(tail_node)
        if not cur_node.is_leaf():
            for children_node in cur_node.children:
                is_output_visible = self.is_node_visible(children_node)
                if is_output_visible:
                    if is_main_node_visible:
                        self.edge_list.append((cur_node, children_node))
                    elif is_tail_node_visible:
                        self.edge_list.append((tail_node, children_node))

        # {tail -> cur_node} part
        # # output node
        # visible tensor and non-input tensor nodes
        if is_cur_visible and not cur_node.is_root():
            assert not isinstance(tail_node, TensorNode) or tail_node.is_root(), (
                "get_tail_node function returned inconsistent Node, please report this"
            )
            self.edge_list.append((tail_node, cur_node))

    def rollify(self, cur_node: ModuleNode | FunctionNode) -> None:
        '''Rolls computational graph by identifying recursively used
        Modules. This is done by giving the same id for nodes that are
        recursively used.
        This becomes complex when there are stateless and torch.functions.
        For more details see docs'''

        head_node = next(iter(cur_node.output_nodes))
        if not head_node.is_leaf() and self.hide_inner_tensors:
            head_node = next(iter(head_node.children))

        # identify recursively used modules
        # with the same node id
        output_id = get_output_id(head_node)
        cur_node.set_node_id(output_id=output_id)

    def is_node_visible(self, compute_node: COMPUTATION_NODES) -> bool:
        '''Returns True if node should be displayed on the visual
        graph. Otherwise False'''

        assert_input_type(
            'is_node_visible', (TensorNode, ModuleNode, FunctionNode,), compute_node
        )

        if compute_node.name == 'empty-pass':
            return False

        if isinstance(compute_node, (ModuleNode, FunctionNode)):
            is_visible = (
                isinstance(compute_node, FunctionNode) or (
                    (self.hide_module_functions and compute_node.is_container)
                    or compute_node.depth == self.depth
                )
            )
            return is_visible

        else:
            if compute_node.main_node.depth < 0 or compute_node.is_aux:
                return False

            is_main_input_or_output = (
                (compute_node.is_root() or compute_node.is_leaf())
                and compute_node.depth == 0
            )
            is_visible = (
                not self.hide_inner_tensors or is_main_input_or_output
            )

            return is_visible

    def get_tail_node(self, _tensor_node: TensorNode) -> COMPUTATION_NODES:

        tensor_node = _tensor_node.main_node if _tensor_node.is_aux else _tensor_node

        # non-output nodes eminating from input node
        if tensor_node.is_root():
            return tensor_node

        current_parent_h = tensor_node.parent_hierarchy

        sorted_depth = sorted(depth for depth in current_parent_h)
        tail_node = next(iter(tensor_node.parents))
        depth = 0
        for depth in sorted_depth:
            tail_node = current_parent_h[depth]
            if depth >= self.depth:
                break

        module_depth = depth - 1
        # if returned by container module and hide_module_functions
        if (
            isinstance(current_parent_h[depth], FunctionNode) and
            module_depth in tensor_node.parent_hierarchy and self.hide_module_functions
        ):
            if current_parent_h[module_depth].is_container:
                return current_parent_h[module_depth]

        # Even though this is recursive, not harmful for complexity
        # The reason: the (time) complexity ~ O(L^2) where L
        # is the length of CONTINUOUS path along which the same tensor is passed
        # without any operation on it. L is always small since we dont use
        # infinitely big network with infinitely big continuou pass of unchanged
        # tensor. This recursion is necessary e.g. for LDC model
        if tail_node.name == 'empty-pass':
            empty_pass_parent = next(iter((tail_node.parents)))
            assert isinstance(empty_pass_parent, TensorNode), (
                f'{empty_pass_parent} is input of {tail_node}'
                f'and must a be TensorNode'
            )
            return self.get_tail_node(empty_pass_parent)
        return tail_node

    def add_edge(
        self, edge_ids: tuple[int, int], edg_cnt: int
    ) -> None:

        tail_id, head_id = edge_ids
        label = None if edg_cnt == 1 else f' x{edg_cnt}'
        self.visual_graph.edge(f'{tail_id}', f'{head_id}', label=label)

    def add_node(
        self, node: COMPUTATION_NODES, subgraph: Digraph | None = None
    ) -> None:
        '''Adds node to the graphviz with correct id, label and color
        settings. Updates state of running_node_id if node is not
        identified before.'''
        if node.node_id not in self.id_dict:
            self.id_dict[node.node_id] = self.running_node_id
            self.running_node_id += 1
        label = self.get_node_label(node)
        node_color = ComputationGraph.get_node_color(node)

        if subgraph is None:
            subgraph = self.visual_graph
        subgraph.node(
            name=f'{self.id_dict[node.node_id]}', label=label, fillcolor=node_color,
        )
        self.node_set.add(id(node))

    def get_node_label(self, node: COMPUTATION_NODES) -> str:
        '''Returns html-like format for the label of node. This html-like
        label is based on Graphviz API for html-like format. For setting of node label
        it uses graph config and html_config.'''
        input_str = 'input'
        output_str = 'output'
        border = self.html_config['border']
        cell_sp = self.html_config['cell_spacing']
        cell_pad = self.html_config['cell_padding']
        cell_bor = self.html_config['cell_border']
        if self.show_shapes:
            if isinstance(node, TensorNode):
                label = f'''<
                    <TABLE BORDER="{border}" CELLBORDER="{cell_bor}"
                    CELLSPACING="{cell_sp}" CELLPADDING="{cell_pad}">
                        <TR><TD>{node.name}<BR/>depth:{node.depth}</TD><TD>{node.get_tensor_shape()}</TD></TR>
                    </TABLE>>'''
            else:
                input_repr = compact_list_repr(node.input_shape)
                output_repr = compact_list_repr(node.output_shape)
                label = f'''<
                    <TABLE BORDER="{border}" CELLBORDER="{cell_bor}"
                    CELLSPACING="{cell_sp}" CELLPADDING="{cell_pad}">
                    <TR>
                        <TD ROWSPAN="2">{node.name}<BR/>depth:{node.depth}</TD>
                        <TD COLSPAN="2">{input_str}:</TD>
                        <TD COLSPAN="2">{input_repr} </TD>
                    </TR>
                    <TR>
                        <TD COLSPAN="2">{output_str}: </TD>
                        <TD COLSPAN="2">{output_repr} </TD>
                    </TR>
                    </TABLE>>'''
        else:
            label = f'''<
                    <TABLE BORDER="{border}" CELLBORDER="{cell_bor}"
                    CELLSPACING="{cell_sp}" CELLPADDING="{cell_pad}">
                        <TR><TD>{node.name}<BR/>depth:{node.depth}</TD></TR>
                    </TABLE>>'''
        return label

    def resize_graph(
        self,
        scale: float = 1.0,
        size_per_element: float = 0.3,
        min_size: float = 12
    ) -> None:
        """Resize the graph according to how much content it contains.
        Modify the graph in place. Default values are subject to change,
        so far they seem to work fine.
        """
        # Get the approximate number of nodes and edges
        num_rows = len(self.visual_graph.body)
        content_size = num_rows * size_per_element
        size = scale * max(min_size, content_size)
        size_str = str(size) + "," + str(size)
        self.visual_graph.graph_attr.update(size=size_str,)

    def to_networkx(self) -> nx.DiGraph:
        """Convert the computation graph to a NetworkX DiGraph.
        
        Creates a hierarchical NetworkX graph that exactly matches the graphviz output:
        - Only includes visible nodes (respects hide_inner_tensors, hide_module_functions)
        - Hierarchical structure is preserved via 'subgraph' attribute
        - Edge information matches the visual graph exactly
        - Rolled modules share the same node ID (matching graphviz behavior)
        
        Returns:
            nx.DiGraph: A NetworkX directed graph matching the graphviz output.
            
        Node attributes include:
            - node_type: 'tensor', 'module', or 'function'
            - name: The node's display name
            - depth: The node's depth in the hierarchy
            - subgraph: The subgraph/group this node belongs to (for hierarchy)
            - subgraph_label: Label of the containing subgraph
            - For TensorNode: tensor_shape, tensor_data (if stored), is_input, is_output
            - For ModuleNode/FunctionNode: input_shape, output_shape, is_container, 
              type_name (for modules), attributes (if collected)
        
        Edge attributes include:
            - count: Number of times this edge appears (for rolled graphs, shown as "x{count}")
        """
        G = nx.DiGraph()
        
        # Use the same id_dict that graphviz uses - this handles rolled modules correctly
        # node.node_id -> integer id (rolled modules have same node_id after rollify())
        # Maps integer id -> networkx node string id
        int_id_to_nx_id: dict[int, str] = {}
        
        # Track subgraph membership
        subgraph_stack: list[tuple[str, str]] = []  # [(subgraph_id, subgraph_label), ...]
        # Track all subgraphs for hierarchy
        subgraphs: dict[str, dict[str, Any]] = {}  # subgraph_id -> {label, parent, children}
        
        def get_nx_node_id(node: COMPUTATION_NODES) -> str | None:
            """Get NetworkX node ID using the same mapping as graphviz."""
            # Use id_dict which is set by graphviz rendering (handles rolled modules)
            if node.node_id not in self.id_dict:
                return None
            
            int_id = self.id_dict[node.node_id]
            if int_id in int_id_to_nx_id:
                return int_id_to_nx_id[int_id]
            
            # Create readable ID based on node type and name
            base_id = f"{node.name}_{int_id}"
            int_id_to_nx_id[int_id] = base_id
            return base_id
        
        def get_node_attributes(node: COMPUTATION_NODES) -> dict[str, Any]:
            """Extract all relevant attributes from a computation node."""
            attrs: dict[str, Any] = {
                'name': node.name,
                'depth': node.depth,
                'internal_id': node.node_id,
                'graphviz_id': self.id_dict.get(node.node_id),
            }
            
            if isinstance(node, TensorNode):
                attrs['node_type'] = 'tensor'
                attrs['tensor_shape'] = node.get_tensor_shape()
                attrs['is_input'] = node.is_root()
                attrs['is_output'] = node.is_leaf()
                attrs['is_aux'] = node.is_aux
                if node.has_tensor_data():
                    attrs['tensor_data'] = node.tensor_data
            
            elif isinstance(node, ModuleNode):
                attrs['node_type'] = 'module'
                attrs['type_name'] = node.type_name
                attrs['input_shape'] = node.input_shape
                attrs['output_shape'] = node.output_shape
                attrs['is_container'] = node.is_container
                attrs['is_activation'] = node.is_activation
                attrs['compute_unit_id'] = node.compute_unit_id
                if node.attributes:
                    attrs['attributes'] = node.attributes
            
            elif isinstance(node, FunctionNode):
                attrs['node_type'] = 'function'
                attrs['input_shape'] = node.input_shape
                attrs['output_shape'] = node.output_shape
                attrs['compute_unit_id'] = node.compute_unit_id
                if node.attributes:
                    attrs['attributes'] = node.attributes
            
            # Add subgraph info
            if subgraph_stack:
                attrs['subgraph'] = subgraph_stack[-1][0]
                attrs['subgraph_label'] = subgraph_stack[-1][1]
            else:
                attrs['subgraph'] = None
                attrs['subgraph_label'] = None
            
            return attrs
        
        def traverse_for_networkx(
            cur_node: COMPUTATION_NODES | dict[ModuleNode, list[Any]],
        ) -> None:
            """Traverse graph following the same logic as graphviz rendering."""
            
            if isinstance(cur_node, (TensorNode, ModuleNode, FunctionNode)):
                if cur_node.depth <= self.depth:
                    # Check if node is visible (same logic as collect_graph)
                    is_isolated = cur_node.is_root() and cur_node.is_leaf()
                    if not is_isolated:
                        if self.is_node_visible(cur_node):
                            nx_id = get_nx_node_id(cur_node)
                            if nx_id and nx_id not in G:
                                attrs = get_node_attributes(cur_node)
                                G.add_node(nx_id, **attrs)
                return
            
            if isinstance(cur_node, dict):
                k, v = list(cur_node.items())[0]
                
                # Process the container module node
                if k.depth <= self.depth and k.depth >= 0:
                    is_isolated = k.is_root() and k.is_leaf()
                    if not is_isolated:
                        if self.is_node_visible(k):
                            nx_id = get_nx_node_id(k)
                            if nx_id and nx_id not in G:
                                attrs = get_node_attributes(k)
                                G.add_node(nx_id, **attrs)
                
                # Skip to outputs if container module with hidden functions
                if self.hide_module_functions and k.is_container:
                    for g in k.output_nodes:
                        traverse_for_networkx(g)
                    return
                
                # Check if we should create a subgraph (nested display)
                display_nested = (
                    k.depth < self.depth and k.depth >= 1 and self.expand_nested
                )
                
                if display_nested:
                    subgraph_id = f"cluster_{k.node_id}"
                    subgraph_label = f"{k.type_name}: {k.name}"
                    parent_subgraph = subgraph_stack[-1][0] if subgraph_stack else None
                    
                    # Only add if not already present (handles rolled modules)
                    if subgraph_id not in subgraphs:
                        subgraphs[subgraph_id] = {
                            'label': subgraph_label,
                            'parent': parent_subgraph,
                            'module_name': k.name,
                            'module_type': k.type_name,
                            'depth': k.depth,
                        }
                    
                    subgraph_stack.append((subgraph_id, subgraph_label))
                
                # Process children
                for g in v:
                    traverse_for_networkx(g)
                
                if display_nested:
                    subgraph_stack.pop()
        
        # Traverse the hierarchy to collect visible nodes
        traverse_for_networkx(self.node_hierarchy)
        
        # Add edges from edge_list using the same id mapping as graphviz
        # This handles rolled modules correctly - same node_id = same edge endpoint
        edge_counter: dict[tuple[str, str], int] = {}
        for tail, head in self.edge_list:
            # Use graphviz id_dict to get integer IDs (handles rolled modules)
            tail_int_id = self.id_dict.get(tail.node_id)
            head_int_id = self.id_dict.get(head.node_id)
            
            if tail_int_id is None or head_int_id is None:
                continue
            
            tail_nx_id = int_id_to_nx_id.get(tail_int_id)
            head_nx_id = int_id_to_nx_id.get(head_int_id)
            
            if tail_nx_id and head_nx_id and tail_nx_id in G and head_nx_id in G:
                edge_key = (tail_nx_id, head_nx_id)
                edge_counter[edge_key] = edge_counter.get(edge_key, 0) + 1
        
        # Add edges with final counts (matching graphviz which shows "x{count}")
        for (tail_nx_id, head_nx_id), count in edge_counter.items():
            G.add_edge(tail_nx_id, head_nx_id, count=count)
        
        # Store subgraph hierarchy and settings as graph attributes
        G.graph['subgraphs'] = subgraphs
        G.graph['settings'] = {
            'show_shapes': self.show_shapes,
            'expand_nested': self.expand_nested,
            'hide_inner_tensors': self.hide_inner_tensors,
            'hide_module_functions': self.hide_module_functions,
            'depth': self.depth,
            'roll': self.roll,
        }
        
        return G
    
    def to_html(self, filename: str | None = None) -> str:
        """Export the computation graph as an interactive HTML file.
        
        Creates an HTML file with:
        - Interactive graph visualization using vis.js
        - Drag and drop positioning of nodes
        - Expand/collapse of hierarchical groups
        - Zoom and pan navigation
        - Node details on hover/click
        
        Args:
            filename: Path to save the HTML file. If None, returns HTML string only.
            
        Returns:
            str: The HTML content as a string.
        """
        import json
        
        # Get the networkx graph
        G = self.to_networkx()
        
        # Prepare nodes data for vis.js
        nodes_data = []
        for node_id, attrs in G.nodes(data=True):
            node_type = attrs.get('node_type', 'unknown')
            name = attrs.get('name', node_id)
            depth = attrs.get('depth', 0)
            
            # Colors matching torchview
            colors = {
                'tensor': {'background': '#FFFFE0', 'border': '#DAA520'},
                'module': {'background': '#8FBC8B', 'border': '#2E8B57'},
                'function': {'background': '#F0F8FF', 'border': '#4169E1'},
            }
            color = colors.get(node_type, {'background': '#FFFFFF', 'border': '#000000'})
            
            # Build label
            if node_type == 'tensor':
                shape = attrs.get('tensor_shape', ())
                label = f"{name}\ndepth:{depth}\n{shape}"
            elif node_type in ('module', 'function'):
                in_shape = attrs.get('input_shape', [])
                out_shape = attrs.get('output_shape', [])
                in_str = str(in_shape[0]) if len(in_shape) == 1 else str(in_shape) if in_shape else ''
                out_str = str(out_shape[0]) if len(out_shape) == 1 else str(out_shape) if out_shape else ''
                if in_str and out_str:
                    label = f"{name}\ndepth:{depth}\nin: {in_str}\nout: {out_str}"
                else:
                    label = f"{name}\ndepth:{depth}"
            else:
                label = name
            
            # Determine group (subgraph)
            group = attrs.get('subgraph', 'root')
            
            node_data = {
                'id': node_id,
                'label': label,
                'color': color,
                'shape': 'box',
                'font': {'face': 'monospace', 'size': 12},
                'group': group if group else 'root',
                'title': json.dumps(attrs, default=str, indent=2),  # Tooltip
                'nodeType': node_type,
                'depth': depth,
            }
            nodes_data.append(node_data)
        
        # Prepare edges data for vis.js
        edges_data = []
        for source, target, attrs in G.edges(data=True):
            count = attrs.get('count', 1)
            edge_data = {
                'from': source,
                'to': target,
                'arrows': 'to',
                'color': {'color': '#666666'},
            }
            if count > 1:
                edge_data['label'] = f'x{count}'
            edges_data.append(edge_data)
        
        # Prepare groups (subgraphs) for clustering
        subgraphs = G.graph.get('subgraphs', {})
        groups_data = {'root': {'color': {'background': '#FAFAFA'}}}
        for sg_id, sg_info in subgraphs.items():
            groups_data[sg_id] = {
                'color': {'background': '#F5F5F5', 'border': '#CCCCCC'},
            }
        
        # Generate HTML
        html_content = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>TorchView - Interactive Computation Graph</title>
    <script src="https://unpkg.com/vis-data@7.1.9/peer/umd/vis-data.min.js"></script>
    <script src="https://unpkg.com/vis-network@9.1.9/peer/umd/vis-network.min.js"></script>
    <link href="https://unpkg.com/vis-network@9.1.9/styles/vis-network.min.css" rel="stylesheet">
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: #1a1a2e;
            color: #eee;
        }}
        #container {{
            display: flex;
            height: 100vh;
        }}
        #graph-container {{
            flex: 1;
            background: #16213e;
            position: relative;
        }}
        #network {{
            width: 100%;
            height: 100%;
        }}
        #sidebar {{
            width: 320px;
            background: #0f3460;
            padding: 20px;
            overflow-y: auto;
            border-left: 2px solid #e94560;
        }}
        h1 {{
            font-size: 1.4em;
            margin-bottom: 15px;
            color: #e94560;
            border-bottom: 1px solid #e94560;
            padding-bottom: 10px;
        }}
        h2 {{
            font-size: 1.1em;
            margin: 15px 0 10px;
            color: #00fff5;
        }}
        .info-section {{
            background: #1a1a2e;
            border-radius: 8px;
            padding: 12px;
            margin-bottom: 15px;
        }}
        .stat {{
            display: flex;
            justify-content: space-between;
            padding: 5px 0;
            border-bottom: 1px solid #333;
        }}
        .stat:last-child {{
            border-bottom: none;
        }}
        .stat-label {{
            color: #888;
        }}
        .stat-value {{
            color: #00fff5;
            font-weight: bold;
        }}
        #node-details {{
            font-family: monospace;
            font-size: 11px;
            white-space: pre-wrap;
            background: #1a1a2e;
            padding: 10px;
            border-radius: 8px;
            max-height: 300px;
            overflow-y: auto;
        }}
        .legend {{
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
            margin-top: 10px;
        }}
        .legend-item {{
            display: flex;
            align-items: center;
            gap: 5px;
            font-size: 12px;
        }}
        .legend-color {{
            width: 20px;
            height: 20px;
            border-radius: 4px;
            border: 2px solid;
        }}
        .controls {{
            margin-top: 15px;
        }}
        button {{
            background: #e94560;
            color: white;
            border: none;
            padding: 8px 16px;
            border-radius: 5px;
            cursor: pointer;
            margin: 5px 5px 5px 0;
            font-size: 12px;
            transition: background 0.3s;
        }}
        button:hover {{
            background: #ff6b6b;
        }}
        .group-list {{
            max-height: 200px;
            overflow-y: auto;
        }}
        .group-item {{
            padding: 5px 10px;
            margin: 3px 0;
            background: #1a1a2e;
            border-radius: 4px;
            cursor: pointer;
            font-size: 12px;
            transition: background 0.2s;
        }}
        .group-item:hover {{
            background: #2a2a4e;
        }}
        .group-item.collapsed {{
            opacity: 0.6;
        }}
    </style>
</head>
<body>
    <div id="container">
        <div id="graph-container">
            <div id="network"></div>
        </div>
        <div id="sidebar">
            <h1>🔥 TorchView Graph</h1>
            
            <div class="info-section">
                <h2>📊 Statistics</h2>
                <div class="stat">
                    <span class="stat-label">Total Nodes</span>
                    <span class="stat-value" id="node-count">{G.number_of_nodes()}</span>
                </div>
                <div class="stat">
                    <span class="stat-label">Total Edges</span>
                    <span class="stat-value" id="edge-count">{G.number_of_edges()}</span>
                </div>
                <div class="stat">
                    <span class="stat-label">Subgraphs</span>
                    <span class="stat-value">{len(subgraphs)}</span>
                </div>
            </div>
            
            <div class="info-section">
                <h2>🎨 Legend</h2>
                <div class="legend">
                    <div class="legend-item">
                        <div class="legend-color" style="background: #FFFFE0; border-color: #DAA520;"></div>
                        <span>Tensor</span>
                    </div>
                    <div class="legend-item">
                        <div class="legend-color" style="background: #8FBC8B; border-color: #2E8B57;"></div>
                        <span>Module</span>
                    </div>
                    <div class="legend-item">
                        <div class="legend-color" style="background: #F0F8FF; border-color: #4169E1;"></div>
                        <span>Function</span>
                    </div>
                </div>
            </div>
            
            <div class="info-section">
                <h2>🎮 Controls</h2>
                <div class="controls">
                    <button onclick="fitNetwork()">Fit to Screen</button>
                    <button onclick="resetLayout()">Reset Layout</button>
                    <button onclick="togglePhysics()">Toggle Physics</button>
                </div>
            </div>
            
            <div class="info-section">
                <h2>📁 Module Groups</h2>
                <div class="group-list" id="group-list"></div>
            </div>
            
            <div class="info-section">
                <h2>📋 Node Details</h2>
                <div id="node-details">Click on a node to see details...</div>
            </div>
        </div>
    </div>

    <script>
        // Graph data
        const nodesData = {json.dumps(nodes_data)};
        const edgesData = {json.dumps(edges_data)};
        const subgraphs = {json.dumps(subgraphs)};
        
        // Create vis.js datasets
        const nodes = new vis.DataSet(nodesData);
        const edges = new vis.DataSet(edgesData);
        
        // Network options
        const options = {{
            layout: {{
                hierarchical: {{
                    enabled: true,
                    direction: 'UD',
                    sortMethod: 'directed',
                    levelSeparation: 100,
                    nodeSpacing: 150,
                    treeSpacing: 200,
                    blockShifting: true,
                    edgeMinimization: true,
                    parentCentralization: true,
                }}
            }},
            physics: {{
                enabled: false,
            }},
            interaction: {{
                hover: true,
                tooltipDelay: 200,
                multiselect: true,
                navigationButtons: true,
            }},
            edges: {{
                smooth: {{
                    type: 'cubicBezier',
                    forceDirection: 'vertical',
                }},
            }},
            nodes: {{
                borderWidth: 2,
                shadow: true,
            }},
        }};
        
        // Create network
        const container = document.getElementById('network');
        const network = new vis.Network(container, {{ nodes, edges }}, options);
        
        // Track physics state
        let physicsEnabled = false;
        
        // Populate group list
        const groupList = document.getElementById('group-list');
        const collapsedGroups = new Set();
        
        Object.entries(subgraphs).forEach(([sgId, sgInfo]) => {{
            const div = document.createElement('div');
            div.className = 'group-item';
            div.textContent = sgInfo.label;
            div.onclick = () => toggleGroup(sgId, div);
            groupList.appendChild(div);
        }});
        
        // Node click handler
        network.on('click', function(params) {{
            if (params.nodes.length > 0) {{
                const nodeId = params.nodes[0];
                const node = nodes.get(nodeId);
                const details = JSON.parse(node.title);
                document.getElementById('node-details').textContent = 
                    JSON.stringify(details, null, 2);
            }}
        }});
        
        // Functions
        function fitNetwork() {{
            network.fit({{ animation: true }});
        }}
        
        function resetLayout() {{
            // Re-enable hierarchical layout temporarily
            network.setOptions({{
                layout: {{
                    hierarchical: {{
                        enabled: true,
                    }}
                }}
            }});
            setTimeout(() => {{
                network.setOptions({{
                    layout: {{
                        hierarchical: {{
                            enabled: false,
                        }}
                    }}
                }});
            }}, 100);
        }}
        
        function togglePhysics() {{
            physicsEnabled = !physicsEnabled;
            network.setOptions({{
                physics: {{ enabled: physicsEnabled }}
            }});
        }}
        
        function toggleGroup(groupId, element) {{
            const groupNodes = nodesData.filter(n => n.group === groupId).map(n => n.id);
            
            if (collapsedGroups.has(groupId)) {{
                // Expand
                collapsedGroups.delete(groupId);
                element.classList.remove('collapsed');
                groupNodes.forEach(nodeId => {{
                    nodes.update({{ id: nodeId, hidden: false }});
                }});
            }} else {{
                // Collapse
                collapsedGroups.add(groupId);
                element.classList.add('collapsed');
                groupNodes.forEach(nodeId => {{
                    nodes.update({{ id: nodeId, hidden: true }});
                }});
            }}
        }}
        
        // Initial fit
        network.once('stabilizationIterationsDone', function() {{
            network.fit();
        }});
    </script>
</body>
</html>'''
        
        if filename:
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(html_content)
        
        return html_content

    @staticmethod
    def get_node_color(
        node: COMPUTATION_NODES
    ) -> str:
        return node2color[type(node)]

    def check_node(self, node: COMPUTATION_NODES) -> None:
        assert node.node_id != 'null', f'wrong id {node} {type(node)}'
        assert '-' not in node.node_id, 'No repetition of node recording is allowed'
        assert node.depth <= self.depth, f"Exceeds display depth limit, {node}"
        assert (
            sum(1 for _ in node.parents) in [0, 1] or not isinstance(node, TensorNode)
        ), (
            f'tensor must have single input node {node}'
        )


def compact_list_repr(x: list[Any]) -> str:
    '''returns more compact representation of list with
    repeated elements. This is useful for e.g. output of transformer/rnn
    models where hidden state outputs shapes is repetation of one hidden unit
    output'''

    list_counter = Counter(x)
    x_repr = ''

    for elem, cnt in list_counter.items():
        if cnt == 1:
            x_repr += f'{elem}, '
        else:
            x_repr += f'{cnt} x {elem}, '

    # get rid of last comma
    return x_repr[:-2]


def get_output_id(head_node: COMPUTATION_NODES) -> str:
    ''' This returns id of output to get correct id.
    This is used to identify the recursively used modules.
    Identification relation is as follows:
        ModuleNodes => by id of nn.Module object
        Parameterless ModulesNodes => by id of nn.Module object
        FunctionNodes => by id of Node object
    '''
    if isinstance(head_node, ModuleNode):
        output_id = str(head_node.compute_unit_id)
    else:
        output_id = head_node.node_id

    return output_id
