#!/usr/bin/env python
"""
TorchView Feature Showcase
==========================

This script demonstrates all the features of the torchview library for
visualizing PyTorch model architectures and computation graphs.

Features demonstrated:
1. Basic graph visualization
2. Input methods (input_data vs input_size)
3. Shape visualization (show_shapes)
4. Nested module visualization (expand_nested)
5. Module function visibility (hide_module_functions)
6. Inner tensor visibility (hide_inner_tensors)
7. Recursive module rolling (roll)
8. Graph direction (graph_dir)
9. Attribute collection (collect_attributes)
10. Tensor data storage (store_tensor_data) - NEW
11. Memory limits (max_tensor_bytes) - NEW
12. Module name mapping (full path names)
13. Graph export and saving
14. Multiple inputs handling
15. Programmatic graph traversal
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchview import draw_graph
from torchview.computation_node import ModuleNode, FunctionNode, TensorNode
from torchview.computation_graph import ComputationGraph
from pathlib import Path

# Create output directory
OUTPUT_DIR = Path(__file__).parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)


# =============================================================================
# Example Models
# =============================================================================

class BranchedMLP(nn.Module):
    """A branched Multi-Layer Perceptron with parallel paths."""
    
    def __init__(self):
        super().__init__()
        
        # Common layers before branching
        self.seq = nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU()
        )
        
        self.inner_seq_1 = nn.Sequential(
            nn.Linear(15, 15),
            nn.ReLU()
        )
        self.inner_seq_2 = nn.Sequential(
            nn.Linear(15, 15),
            nn.ReLU()
        )
        
        # Branch 1
        self.branch1 = nn.Sequential(
            nn.Linear(20, 15),
            self.inner_seq_1,
            self.inner_seq_2
        )
        
        # Branch 2
        self.branch2 = nn.Sequential(
            nn.Linear(20, 15),
            nn.ReLU()
        )
        
        # Common layers after recombining
        self.fc_combine = nn.Linear(30, 10)
        self.fc_output = nn.Linear(10, 1)
        
        # ModuleList after recombination
        self.module_list = nn.ModuleList([
            nn.Linear(1, 5),
            nn.ReLU(),
            nn.Linear(5, 1)
        ])
        
        self.branch_output = nn.Linear(5, 5)
        self.branch_logit = nn.ReLU()

    def forward(self, x):
        # Common forward pass before branching
        x = self.seq(x)
        
        # Branching
        branch1_out = self.branch1(x)
        branch2_out = self.branch2(x)
        
        # Concatenate the outputs of both branches
        x = torch.cat([branch1_out, branch2_out], dim=1)
        
        # Pass through the combined layer
        x = self.fc_combine(x)
        x = self.fc_output(x)
        
        # ModuleList forward pass
        for i, layer in enumerate(self.module_list):
            x = layer(x)
            if i == 1:
                x2 = self.branch_output(x)
                x2 = self.branch_logit(x2)
        
        return x, x2


class ConvNet(nn.Module):
    """A Convolutional Neural Network for image classification."""
    
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 10),
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


class RecursiveBlock(nn.Module):
    """A block that is used recursively."""
    
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class RecursiveNet(nn.Module):
    """A network that uses the same block multiple times."""
    
    def __init__(self, num_repeats=3):
        super().__init__()
        self.input_conv = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.block = RecursiveBlock()
        self.num_repeats = num_repeats
        self.output_conv = nn.Conv2d(64, 10, kernel_size=1)
    
    def forward(self, x):
        x = self.input_conv(x)
        for _ in range(self.num_repeats):
            x = self.block(x)
        x = self.output_conv(x)
        return x


class SiameseNetwork(nn.Module):
    """A Siamese network that processes two inputs."""
    
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(1024, 256),
        )
        self.fc = nn.Linear(256, 1)
    
    def forward(self, x1, x2):
        # Process both inputs through the same encoder
        feat1 = self.encoder(x1)
        feat2 = self.encoder(x2)
        # Compute similarity
        diff = torch.abs(feat1 - feat2)
        out = torch.sigmoid(self.fc(diff))
        return out


class AttentionBlock(nn.Module):
    """Simple self-attention block."""
    
    def __init__(self, embed_dim=64, num_heads=4):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim),
        )
        self.norm2 = nn.LayerNorm(embed_dim)
    
    def forward(self, x):
        # Self-attention with residual
        attn_out, _ = self.attention(x, x, x)
        x = self.norm(x + attn_out)
        # FFN with residual
        x = self.norm2(x + self.ffn(x))
        return x


class TransformerModel(nn.Module):
    """A simple Transformer-style model."""
    
    def __init__(self, vocab_size=1000, embed_dim=64, num_layers=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoding = nn.Parameter(torch.randn(1, 100, embed_dim))
        self.layers = nn.ModuleList([
            AttentionBlock(embed_dim) for _ in range(num_layers)
        ])
        self.output = nn.Linear(embed_dim, vocab_size)
    
    def forward(self, x):
        x = self.embedding(x) + self.pos_encoding[:, :x.size(1), :]
        for layer in self.layers:
            x = layer(x)
        return self.output(x)


# =============================================================================
# Feature Demonstrations
# =============================================================================

def demo_basic_visualization():
    """1. Basic graph visualization."""
    print("\n" + "="*60)
    print("1. BASIC GRAPH VISUALIZATION")
    print("="*60)
    
    model = BranchedMLP()
    graph = draw_graph(
        model,
        input_size=(1, 10),
        graph_name="BranchedMLP",
    )
    
    print(f"✓ Created graph for BranchedMLP")
    print(f"  - Number of edges: {len(graph.edge_list)}")
    print(f"  - Number of nodes in ID dict: {len(graph.id_dict)}")
    
    # Save the graph
    graph.visual_graph.render(
        filename=str(OUTPUT_DIR / "1_basic_mlp"),
        format="png",
        cleanup=True
    )
    print(f"  - Saved to: {OUTPUT_DIR / '1_basic_mlp.png'}")
    
    return graph


def demo_input_methods():
    """2. Different input methods."""
    print("\n" + "="*60)
    print("2. INPUT METHODS (input_size vs input_data)")
    print("="*60)
    
    model = BranchedMLP()
    
    # Method 1: Using input_size
    graph1 = draw_graph(
        model,
        input_size=(1, 10),
        graph_name="MLP_input_size",
    )
    print("✓ Created graph using input_size=(1, 10)")
    
    # Method 2: Using input_data (actual tensor)
    input_tensor = torch.randn(2, 10)
    graph2 = draw_graph(
        model,
        input_data=input_tensor,
        graph_name="MLP_input_data",
    )
    print(f"✓ Created graph using input_data with shape {tuple(input_tensor.shape)}")
    
    return graph1, graph2


def demo_show_shapes():
    """3. Shape visualization toggle."""
    print("\n" + "="*60)
    print("3. SHAPE VISUALIZATION (show_shapes)")
    print("="*60)
    
    model = ConvNet()
    
    # With shapes
    graph_with_shapes = draw_graph(
        model,
        input_size=(1, 3, 32, 32),
        graph_name="ConvNet_with_shapes",
        show_shapes=True,
    )
    graph_with_shapes.visual_graph.render(
        filename=str(OUTPUT_DIR / "3a_convnet_with_shapes"),
        format="png",
        cleanup=True
    )
    print("✓ Created graph WITH shapes → 3a_convnet_with_shapes.png")
    
    # Without shapes
    graph_no_shapes = draw_graph(
        model,
        input_size=(1, 3, 32, 32),
        graph_name="ConvNet_no_shapes",
        show_shapes=False,
    )
    graph_no_shapes.visual_graph.render(
        filename=str(OUTPUT_DIR / "3b_convnet_no_shapes"),
        format="png",
        cleanup=True
    )
    print("✓ Created graph WITHOUT shapes → 3b_convnet_no_shapes.png")
    
    return graph_with_shapes, graph_no_shapes


def demo_expand_nested():
    """4. Nested module visualization."""
    print("\n" + "="*60)
    print("4. NESTED MODULE VISUALIZATION (expand_nested)")
    print("="*60)
    
    # --- ConvNet examples ---
    model = ConvNet()
    
    # Collapsed (default)
    graph_collapsed = draw_graph(
        model,
        input_size=(1, 3, 32, 32),
        graph_name="ConvNet_collapsed",
        expand_nested=False,
        depth=3,
    )
    graph_collapsed.visual_graph.render(
        filename=str(OUTPUT_DIR / "4a_convnet_collapsed"),
        format="png",
        cleanup=True
    )
    print("✓ Created COLLAPSED ConvNet graph → 4a_convnet_collapsed.png")
    
    # Expanded
    graph_expanded = draw_graph(
        model,
        input_size=(1, 3, 32, 32),
        graph_name="ConvNet_expanded",
        expand_nested=True,
        depth=10,
    )
    graph_expanded.visual_graph.render(
        filename=str(OUTPUT_DIR / "4b_convnet_expanded"),
        format="png",
        cleanup=True
    )
    print("✓ Created EXPANDED ConvNet graph → 4b_convnet_expanded.png")
    
    # --- BranchedMLP examples ---
    mlp_model = BranchedMLP()
    
    # Collapsed
    mlp_collapsed = draw_graph(
        mlp_model,
        input_size=(1, 10),
        graph_name="BranchedMLP_collapsed",
        expand_nested=False,
        depth=3,
    )
    mlp_collapsed.visual_graph.render(
        filename=str(OUTPUT_DIR / "4c_mlp_collapsed"),
        format="png",
        cleanup=True
    )
    print("✓ Created COLLAPSED BranchedMLP graph → 4c_mlp_collapsed.png")
    
    # Expanded
    mlp_expanded = draw_graph(
        mlp_model,
        input_size=(1, 10),
        graph_name="BranchedMLP_expanded",
        expand_nested=True,
        depth=10,
    )
    mlp_expanded.visual_graph.render(
        filename=str(OUTPUT_DIR / "4d_mlp_expanded"),
        format="png",
        cleanup=True
    )
    print("✓ Created EXPANDED BranchedMLP graph → 4d_mlp_expanded.png")
    
    return graph_collapsed, graph_expanded, mlp_collapsed, mlp_expanded


def demo_hide_module_functions():
    """5. Module function visibility."""
    print("\n" + "="*60)
    print("5. MODULE FUNCTION VISIBILITY (hide_module_functions)")
    print("="*60)
    
    model = BranchedMLP()
    
    # Hide module functions (cleaner view)
    graph_hidden = draw_graph(
        model,
        input_size=(1, 10),
        graph_name="MLP_functions_hidden",
        hide_module_functions=True,
    )
    graph_hidden.visual_graph.render(
        filename=str(OUTPUT_DIR / "5a_mlp_functions_hidden"),
        format="png",
        cleanup=True
    )
    print("✓ Created graph with functions HIDDEN → 5a_mlp_functions_hidden.png")
    
    # Show module functions (detailed view)
    graph_shown = draw_graph(
        model,
        input_size=(1, 10),
        graph_name="MLP_functions_shown",
        hide_module_functions=False,
    )
    graph_shown.visual_graph.render(
        filename=str(OUTPUT_DIR / "5b_mlp_functions_shown"),
        format="png",
        cleanup=True
    )
    print("✓ Created graph with functions SHOWN → 5b_mlp_functions_shown.png")
    
    return graph_hidden, graph_shown


def demo_hide_inner_tensors():
    """6. Inner tensor visibility."""
    print("\n" + "="*60)
    print("6. INNER TENSOR VISIBILITY (hide_inner_tensors)")
    print("="*60)
    
    model = BranchedMLP()
    
    # Hide inner tensors (default, cleaner)
    graph_hidden = draw_graph(
        model,
        input_size=(1, 10),
        graph_name="MLP_tensors_hidden",
        hide_inner_tensors=True,
    )
    graph_hidden.visual_graph.render(
        filename=str(OUTPUT_DIR / "6a_mlp_tensors_hidden"),
        format="png",
        cleanup=True
    )
    print("✓ Created graph with inner tensors HIDDEN → 6a_mlp_tensors_hidden.png")
    
    # Show inner tensors (detailed)
    graph_shown = draw_graph(
        model,
        input_size=(1, 10),
        graph_name="MLP_tensors_shown",
        hide_inner_tensors=False,
    )
    graph_shown.visual_graph.render(
        filename=str(OUTPUT_DIR / "6b_mlp_tensors_shown"),
        format="png",
        cleanup=True
    )
    print("✓ Created graph with inner tensors SHOWN → 6b_mlp_tensors_shown.png")
    
    return graph_hidden, graph_shown


def demo_roll_recursive():
    """7. Recursive module rolling."""
    print("\n" + "="*60)
    print("7. RECURSIVE MODULE ROLLING (roll)")
    print("="*60)
    
    model = RecursiveNet(num_repeats=4)
    
    # Unrolled (shows each iteration)
    graph_unrolled = draw_graph(
        model,
        input_size=(1, 3, 32, 32),
        graph_name="RecursiveNet_unrolled",
        roll=False,
    )
    graph_unrolled.visual_graph.render(
        filename=str(OUTPUT_DIR / "7a_recursive_unrolled"),
        format="png",
        cleanup=True
    )
    print("✓ Created UNROLLED graph → 7a_recursive_unrolled.png")
    
    # Rolled (collapses recursive uses)
    graph_rolled = draw_graph(
        model,
        input_size=(1, 3, 32, 32),
        graph_name="RecursiveNet_rolled",
        roll=True,
    )
    graph_rolled.visual_graph.render(
        filename=str(OUTPUT_DIR / "7b_recursive_rolled"),
        format="png",
        cleanup=True
    )
    print("✓ Created ROLLED graph → 7b_recursive_rolled.png")
    
    return graph_unrolled, graph_rolled


def demo_graph_direction():
    """8. Graph direction."""
    print("\n" + "="*60)
    print("8. GRAPH DIRECTION (graph_dir)")
    print("="*60)
    
    model = BranchedMLP()
    
    directions = ['TB', 'LR', 'BT', 'RL']
    direction_names = {
        'TB': 'Top to Bottom',
        'LR': 'Left to Right', 
        'BT': 'Bottom to Top',
        'RL': 'Right to Left'
    }
    
    graphs = {}
    for direction in directions:
        graph = draw_graph(
            model,
            input_size=(1, 10),
            graph_name=f"MLP_{direction}",
            graph_dir=direction,
        )
        graph.visual_graph.render(
            filename=str(OUTPUT_DIR / f"8_{direction.lower()}_direction"),
            format="png",
            cleanup=True
        )
        graphs[direction] = graph
        print(f"✓ Created graph with direction {direction} ({direction_names[direction]}) → 8_{direction.lower()}_direction.png")
    
    return graphs


def demo_collect_attributes():
    """9. Attribute collection."""
    print("\n" + "="*60)
    print("9. ATTRIBUTE COLLECTION (collect_attributes)")
    print("="*60)
    
    model = BranchedMLP()
    
    # With attributes
    graph = draw_graph(
        model,
        input_size=(1, 10),
        graph_name="MLP_with_attributes",
        collect_attributes=True,
        expand_nested=True,
        depth=10,
    )
    
    print("✓ Collected module attributes:")
    
    # Traverse and print attributes
    def print_attributes(graph: ComputationGraph):
        kwargs = {'cur_node': graph.node_hierarchy, 'subgraph': None}
        
        def collect_attrs(**kw):
            node = kw['cur_node']
            if isinstance(node, ModuleNode) and node.attributes:
                print(f"  - {node.name}: {node.attributes[:60]}...")
        
        graph.traverse_graph(collect_attrs, **kwargs)
    
    print_attributes(graph)
    
    return graph


def demo_store_tensor_data():
    """10. Tensor data storage (NEW FEATURE)."""
    print("\n" + "="*60)
    print("10. TENSOR DATA STORAGE (store_tensor_data) - NEW!")
    print("="*60)
    
    model = BranchedMLP()
    
    # Without tensor data (default)
    graph_no_data = draw_graph(
        model,
        input_size=(1, 10),
        graph_name="MLP_no_tensor_data",
        store_tensor_data=False,
    )
    
    # With tensor data
    graph_with_data = draw_graph(
        model,
        input_size=(1, 10),
        graph_name="MLP_with_tensor_data",
        store_tensor_data=True,
    )
    
    print("✓ Comparing tensor storage:")
    print("\n  Without store_tensor_data:")
    for node in graph_no_data.root_container:
        print(f"    - {node.name}: has_tensor_data={node.has_tensor_data()}, shape={node.get_tensor_shape()}")
    
    print("\n  With store_tensor_data=True:")
    for node in graph_with_data.root_container:
        has_data = node.has_tensor_data()
        print(f"    - {node.name}: has_tensor_data={has_data}, shape={node.get_tensor_shape()}")
        if has_data:
            print(f"      tensor_data dtype={node.tensor_data.dtype}, values preview: {node.tensor_data.flatten()[:5]}...")
    
    return graph_no_data, graph_with_data


def demo_max_tensor_bytes():
    """11. Memory limits for tensor storage (NEW FEATURE)."""
    print("\n" + "="*60)
    print("11. MEMORY LIMITS (max_tensor_bytes) - NEW!")
    print("="*60)
    
    model = BranchedMLP()
    
    # Large limit (100MB default)
    graph_large_limit = draw_graph(
        model,
        input_size=(1, 10),
        graph_name="MLP_large_limit",
        store_tensor_data=True,
        max_tensor_bytes=100 * 1024 * 1024,  # 100MB
    )
    
    # Very small limit (only 100 bytes)
    import warnings
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        graph_small_limit = draw_graph(
            model,
            input_size=(1, 10),
            graph_name="MLP_small_limit",
            store_tensor_data=True,
            max_tensor_bytes=100,  # Only 100 bytes
        )
        if w:
            print(f"✓ Warnings generated for tensors exceeding limit:")
            for warning in w[:3]:  # Show first 3
                print(f"    - {warning.message}")
    
    print("\n  With max_tensor_bytes=100MB:")
    for node in graph_large_limit.root_container:
        print(f"    - {node.name}: stored={node.has_tensor_data()}")
    
    print("\n  With max_tensor_bytes=100 bytes (very small):")
    for node in graph_small_limit.root_container:
        print(f"    - {node.name}: stored={node.has_tensor_data()}")
    
    return graph_large_limit, graph_small_limit


def demo_module_name_mapping():
    """12. Module name mapping (shows full path names)."""
    print("\n" + "="*60)
    print("12. MODULE NAME MAPPING (Full Path Names)")
    print("="*60)
    
    model = ConvNet()
    
    graph = draw_graph(
        model,
        input_size=(1, 3, 32, 32),
        graph_name="ConvNet_module_names",
        expand_nested=True,
        depth=10,
    )
    
    print("✓ Module name map (id -> path):")
    # Show some module names
    for module_id, name in list(graph.module_name_map.items())[:10]:
        print(f"    {name or '(root)'}")
    
    graph.visual_graph.render(
        filename=str(OUTPUT_DIR / "12_module_names"),
        format="png",
        cleanup=True
    )
    print(f"\n  Saved expanded view → 12_module_names.png")
    
    return graph


def demo_multiple_inputs():
    """14. Multiple inputs handling."""
    print("\n" + "="*60)
    print("14. MULTIPLE INPUTS HANDLING")
    print("="*60)
    
    model = SiameseNetwork()
    
    # Using list of input sizes
    graph = draw_graph(
        model,
        input_size=[(1, 1, 28, 28), (1, 1, 28, 28)],
        graph_name="SiameseNetwork",
        expand_nested=True,
    )
    
    graph.visual_graph.render(
        filename=str(OUTPUT_DIR / "14_siamese_network"),
        format="png",
        cleanup=True
    )
    print("✓ Created graph for Siamese network with 2 inputs → 14_siamese_network.png")
    
    # Using actual tensors
    x1 = torch.randn(1, 1, 28, 28)
    x2 = torch.randn(1, 1, 28, 28)
    graph2 = draw_graph(
        model,
        input_data=[x1, x2],
        graph_name="SiameseNetwork_tensors",
    )
    print("✓ Also works with actual tensor inputs")
    
    return graph, graph2


def demo_programmatic_traversal():
    """15. Programmatic graph traversal."""
    print("\n" + "="*60)
    print("15. PROGRAMMATIC GRAPH TRAVERSAL")
    print("="*60)
    
    model = ConvNet()
    
    graph = draw_graph(
        model,
        input_size=(1, 3, 32, 32),
        graph_name="ConvNet_traversal",
        collect_attributes=True,
    )
    
    # Collect statistics
    modules = []
    functions = []
    tensors = []
    
    def collect_nodes(**kwargs):
        node = kwargs['cur_node']
        if isinstance(node, ModuleNode):
            modules.append({
                'name': node.name,
                'type': node.type_name,
                'input_shape': node.input_shape,
                'output_shape': node.output_shape,
            })
        elif isinstance(node, FunctionNode):
            functions.append({
                'name': node.name,
                'input_shape': node.input_shape,
                'output_shape': node.output_shape,
            })
        elif isinstance(node, TensorNode):
            tensors.append({
                'name': node.name,
                'shape': node.get_tensor_shape(),
            })
    
    kwargs = {'cur_node': graph.node_hierarchy, 'subgraph': None}
    graph.traverse_graph(collect_nodes, **kwargs)
    
    print(f"✓ Graph Statistics:")
    print(f"    - Total modules: {len(modules)}")
    print(f"    - Total functions: {len(functions)}")
    print(f"    - Total tensors: {len(tensors)}")
    print(f"    - Total edges: {len(graph.edge_list)}")
    
    print(f"\n  Sample modules:")
    for m in modules[:5]:
        print(f"    - {m['type']}: {m['name']} {m['input_shape']} → {m['output_shape']}")
    
    return graph, {'modules': modules, 'functions': functions, 'tensors': tensors}


def demo_transformer_model():
    """Bonus: Transformer model visualization."""
    print("\n" + "="*60)
    print("BONUS: TRANSFORMER MODEL VISUALIZATION")
    print("="*60)
    
    model = TransformerModel(vocab_size=1000, embed_dim=64, num_layers=2)
    
    # Input is sequence of token IDs
    graph = draw_graph(
        model,
        input_size=(1, 20),  # batch_size=1, seq_len=20
        dtypes=[torch.long],  # Token IDs are long integers
        graph_name="TransformerModel",
        expand_nested=True,
        depth=5,
    )
    
    graph.visual_graph.render(
        filename=str(OUTPUT_DIR / "bonus_transformer"),
        format="png",
        cleanup=True
    )
    print("✓ Created Transformer model graph → bonus_transformer.png")
    
    return graph


# =============================================================================
# Main
# =============================================================================

def main():
    print("=" * 60)
    print("   TORCHVIEW FEATURE SHOWCASE")
    print("=" * 60)
    print(f"\nOutput directory: {OUTPUT_DIR}")
    
    # Run all demos
    demo_basic_visualization()
    demo_input_methods()
    demo_show_shapes()
    demo_expand_nested()
    demo_hide_module_functions()
    demo_hide_inner_tensors()
    demo_roll_recursive()
    demo_graph_direction()
    demo_collect_attributes()
    demo_store_tensor_data()
    demo_max_tensor_bytes()
    demo_module_name_mapping()
    demo_multiple_inputs()
    demo_programmatic_traversal()
    demo_transformer_model()
    
    print("\n" + "=" * 60)
    print("   ALL DEMOS COMPLETED!")
    print("=" * 60)
    print(f"\nCheck {OUTPUT_DIR} for generated visualizations.")
    print("\nKey new features demonstrated:")
    print("  • store_tensor_data: Store full tensor values for debugging")
    print("  • max_tensor_bytes: Control memory usage for tensor storage")
    print("  • Module name mapping: Full path names like 'features.0' instead of 'Conv2d'")


if __name__ == "__main__":
    main()
