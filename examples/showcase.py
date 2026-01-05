#!/usr/bin/env python
"""
TorchView Feature Showcase - Simplified
=======================================

Demonstrates torchview with a single comprehensive model that includes:
- Convolutional layers
- Recursive blocks
- Branching paths
- Multiple inputs and outputs
- Attention mechanisms (MultiheadAttention)

Outputs:
1. Expanded graph (LR direction) with tensors and functions
2. Expanded graph (TB direction) with tensors and functions
3. Expanded graph (LR direction) without tensors
4. Interactive HTML from NetworkX export
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torchview import draw_graph
from pathlib import Path

# Create output directory
OUTPUT_DIR = Path(__file__).parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)


# =============================================================================
# Comprehensive Model
# =============================================================================

class RecursiveConvBlock(nn.Module):
    """A convolutional block used recursively."""
    
    def __init__(self, channels: int = 64):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class AttentionBranch(nn.Module):
    """Branch with self-attention."""
    
    def __init__(self, embed_dim: int = 64, num_heads: int = 4):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Linear(embed_dim * 2, embed_dim),
        )
    
    def forward(self, x):
        # x: (B, seq, embed_dim)
        attn_out, _ = self.attention(x, x, x)
        x = self.norm(x + attn_out)
        x = x + self.ffn(x)
        return x


class ConvBranch(nn.Module):
    """Convolutional branch with pooling."""
    
    def __init__(self, in_channels: int = 64, out_channels: int = 64):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.pool = nn.AdaptiveAvgPool2d((4, 4))
    
    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        return x


class ComprehensiveModel(nn.Module):
    """
    A comprehensive model combining:
    - ConvNet (input processing)
    - Recursive blocks
    - Branching (conv branch + attention branch)
    - Multiple inputs (image + sequence)
    - Multiple outputs (classification + features)
    - Attention mechanisms
    """
    
    def __init__(
        self,
        num_classes: int = 10,
        num_recursive: int = 3,
        embed_dim: int = 64,
    ):
        super().__init__()
        self.num_recursive = num_recursive
        self.embed_dim = embed_dim
        
        # === Image input processing (ConvNet) ===
        self.input_conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        
        # === Recursive block (applied multiple times) ===
        self.recursive_block = RecursiveConvBlock(channels=64)
        
        # === Branch 1: Convolutional path ===
        self.conv_branch = ConvBranch(in_channels=64, out_channels=64)
        
        # === Branch 2: Attention path (requires flattening) ===
        self.to_sequence = nn.Sequential(
            nn.AdaptiveAvgPool2d((8, 8)),
            nn.Flatten(2),  # (B, C, H*W)
        )
        self.attention_branch = AttentionBranch(embed_dim=64, num_heads=4)
        
        # === Sequence input processing ===
        self.seq_embedding = nn.Linear(32, embed_dim)
        self.seq_attention = AttentionBranch(embed_dim=embed_dim, num_heads=4)
        
        # === Combine branches ===
        # Conv branch output: (B, 64, 4, 4) -> flatten -> (B, 1024)
        # Attention branch output: (B, 64, 64) -> mean -> (B, 64)
        # Seq branch output: (B, seq_len, 64) -> mean -> (B, 64)
        self.conv_flatten = nn.Flatten()
        self.combine = nn.Linear(1024 + 64 + 64, 256)
        
        # === Output heads (multiple outputs) ===
        self.classifier = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )
        self.feature_head = nn.Sequential(
            nn.ReLU(),
            nn.Linear(256, 128),
        )
    
    def forward(self, image, sequence):
        """
        Args:
            image: (B, 3, H, W) - Image input
            sequence: (B, seq_len, 32) - Sequence input
        
        Returns:
            classification: (B, num_classes)
            features: (B, 128)
        """
        # Process image input
        x = self.input_conv(image)
        
        # Apply recursive block multiple times
        for _ in range(self.num_recursive):
            x = self.recursive_block(x)
        
        # Branch 1: Convolutional path
        conv_out = self.conv_branch(x)
        conv_out = self.conv_flatten(conv_out)  # (B, 1024)
        
        # Branch 2: Attention path on image features
        attn_in = self.to_sequence(x)  # (B, 64, 64)
        attn_in = attn_in.transpose(1, 2)  # (B, 64, 64) - treat as (B, seq, embed)
        attn_out = self.attention_branch(attn_in)
        attn_out = attn_out.mean(dim=1)  # (B, 64)
        
        # Process sequence input
        seq_emb = self.seq_embedding(sequence)  # (B, seq_len, 64)
        seq_out = self.seq_attention(seq_emb)
        seq_out = seq_out.mean(dim=1)  # (B, 64)
        
        # Combine all branches
        combined = torch.cat([conv_out, attn_out, seq_out], dim=1)
        combined = self.combine(combined)
        
        # Multiple outputs
        classification = self.classifier(combined)
        features = self.feature_head(combined)
        
        return classification, features


# =============================================================================
# Visualization Functions
# =============================================================================

def create_visualizations():
    """Generate all visualization outputs."""
    print("=" * 60)
    print("   TORCHVIEW COMPREHENSIVE MODEL SHOWCASE")
    print("=" * 60)
    print(f"\nOutput directory: {OUTPUT_DIR}")
    
    model = ComprehensiveModel(num_classes=10, num_recursive=3, embed_dim=64)
    
    # Input sizes: image (B, 3, 32, 32), sequence (B, 16, 32)
    input_sizes = [(1, 3, 32, 32), (1, 16, 32)]
    
    # =========================================================================
    # 1. Expanded graph (LR) with tensors and functions
    # =========================================================================
    print("\n" + "-" * 60)
    print("1. Expanded Graph (Left-to-Right) with Tensors & Functions")
    print("-" * 60)
    
    graph_lr = draw_graph(
        model,
        input_size=input_sizes,
        graph_name="ComprehensiveModel_LR",
        graph_dir="LR",
        expand_nested=True,
        depth=10,
        roll=True,
        show_shapes=True,
        hide_inner_tensors=False,
        hide_module_functions=False,
    )
    graph_lr.visual_graph.render(
        filename=str(OUTPUT_DIR / "1_expanded_lr_with_tensors"),
        format="png",
        cleanup=True
    )
    print(f"✓ Saved: 1_expanded_lr_with_tensors.png")
    
    # =========================================================================
    # 2. Expanded graph (TB) with tensors and functions
    # =========================================================================
    print("\n" + "-" * 60)
    print("2. Expanded Graph (Top-to-Bottom) with Tensors & Functions")
    print("-" * 60)
    
    graph_tb = draw_graph(
        model,
        input_size=input_sizes,
        graph_name="ComprehensiveModel_TB",
        graph_dir="TB",
        expand_nested=True,
        depth=10,
        roll=True,
        show_shapes=True,
        hide_inner_tensors=False,
        hide_module_functions=False,
    )
    graph_tb.visual_graph.render(
        filename=str(OUTPUT_DIR / "2_expanded_tb_with_tensors"),
        format="png",
        cleanup=True
    )
    print(f"✓ Saved: 2_expanded_tb_with_tensors.png")
    
    # =========================================================================
    # 3. Expanded graph (LR) without tensors
    # =========================================================================
    print("\n" + "-" * 60)
    print("3. Expanded Graph (Left-to-Right) without Tensors")
    print("-" * 60)
    
    graph_no_tensors = draw_graph(
        model,
        input_size=input_sizes,
        graph_name="ComprehensiveModel_NoTensors",
        graph_dir="LR",
        expand_nested=True,
        depth=10,
        roll=True,
        show_shapes=True,
        hide_inner_tensors=True,
        hide_module_functions=False,
    )
    graph_no_tensors.visual_graph.render(
        filename=str(OUTPUT_DIR / "3_expanded_lr_no_tensors"),
        format="png",
        cleanup=True
    )
    print(f"✓ Saved: 3_expanded_lr_no_tensors.png")
    
    # =========================================================================
    # 4. Interactive HTML from NetworkX (without draw_graph rendering)
    # =========================================================================
    print("\n" + "-" * 60)
    print("4. Interactive HTML from NetworkX Export")
    print("-" * 60)
    
    # Use the graph we already created to get NetworkX export
    # Then generate HTML solely from the NetworkX data
    nx_graph = graph_lr.to_networkx()
    
    print(f"  NetworkX Graph Statistics:")
    print(f"    - Nodes: {nx_graph.number_of_nodes()}")
    print(f"    - Edges: {nx_graph.number_of_edges()}")
    print(f"    - Subgraphs: {len(nx_graph.graph.get('subgraphs', {}))}")
    
    # Generate HTML directly from NetworkX graph
    html_path = OUTPUT_DIR / "4_interactive_graph.html"
    html_content = generate_html_from_networkx(nx_graph)
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"✓ Saved: {html_path.name}")
    print(f"\n  Open in browser: file://{html_path.absolute()}")
    
    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 60)
    print("   ALL OUTPUTS GENERATED!")
    print("=" * 60)
    print(f"\nFiles created in {OUTPUT_DIR}:")
    print("  1. 1_expanded_lr_with_tensors.png  - Full graph (LR)")
    print("  2. 2_expanded_tb_with_tensors.png  - Full graph (TB)")
    print("  3. 3_expanded_lr_no_tensors.png    - Graph without tensors (LR)")
    print("  4. 4_interactive_graph.html        - Interactive HTML viewer")
    
    return graph_lr, graph_tb, graph_no_tensors, nx_graph


def generate_html_from_networkx(G) -> str:
    """
    Generate interactive HTML visualization solely from NetworkX graph data.
    No dependency on torchview's draw_graph or visual_graph.
    """
    import json
    
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
        
        # Clean attrs for JSON serialization
        clean_attrs = {k: str(v) if not isinstance(v, (str, int, float, bool, list, dict, type(None))) else v 
                       for k, v in attrs.items()}
        
        node_data = {
            'id': node_id,
            'label': label,
            'color': color,
            'shape': 'box',
            'font': {'face': 'monospace', 'size': 11, 'color': '#333'},
            'group': group if group else 'root',
            'title': json.dumps(clean_attrs, default=str, indent=2),
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
    
    # Prepare subgraphs
    subgraphs = G.graph.get('subgraphs', {})
    
    # Count node types
    node_type_counts = {}
    for _, attrs in G.nodes(data=True):
        nt = attrs.get('node_type', 'unknown')
        node_type_counts[nt] = node_type_counts.get(nt, 0) + 1
    
    # Generate HTML
    html_content = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>TorchView - NetworkX Graph Viewer</title>
    <script src="https://unpkg.com/vis-data@7.1.9/peer/umd/vis-data.min.js"></script>
    <script src="https://unpkg.com/vis-network@9.1.9/peer/umd/vis-network.min.js"></script>
    <link href="https://unpkg.com/vis-network@9.1.9/styles/vis-network.min.css" rel="stylesheet">
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: 'JetBrains Mono', 'Fira Code', monospace;
            background: linear-gradient(135deg, #0d1117 0%, #161b22 100%);
            color: #c9d1d9;
            min-height: 100vh;
        }}
        #container {{ display: flex; height: 100vh; }}
        #graph-container {{
            flex: 1;
            background: #0d1117;
            border-right: 1px solid #30363d;
        }}
        #network {{ width: 100%; height: 100%; }}
        #sidebar {{
            width: 340px;
            background: #161b22;
            padding: 20px;
            overflow-y: auto;
        }}
        h1 {{
            font-size: 1.3em;
            margin-bottom: 20px;
            color: #58a6ff;
            display: flex;
            align-items: center;
            gap: 10px;
        }}
        h2 {{
            font-size: 0.9em;
            margin: 18px 0 10px;
            color: #8b949e;
            text-transform: uppercase;
            letter-spacing: 1px;
        }}
        .section {{
            background: #0d1117;
            border: 1px solid #30363d;
            border-radius: 6px;
            padding: 14px;
            margin-bottom: 14px;
        }}
        .stat-row {{
            display: flex;
            justify-content: space-between;
            padding: 6px 0;
            border-bottom: 1px solid #21262d;
        }}
        .stat-row:last-child {{ border-bottom: none; }}
        .stat-label {{ color: #8b949e; font-size: 0.85em; }}
        .stat-value {{ color: #58a6ff; font-weight: 600; }}
        .legend {{
            display: flex;
            flex-wrap: wrap;
            gap: 12px;
        }}
        .legend-item {{
            display: flex;
            align-items: center;
            gap: 6px;
            font-size: 0.85em;
        }}
        .legend-color {{
            width: 16px;
            height: 16px;
            border-radius: 3px;
            border: 2px solid;
        }}
        .controls {{ display: flex; flex-wrap: wrap; gap: 8px; }}
        button {{
            background: #238636;
            color: white;
            border: none;
            padding: 8px 14px;
            border-radius: 6px;
            cursor: pointer;
            font-size: 0.8em;
            font-family: inherit;
            transition: all 0.2s;
        }}
        button:hover {{ background: #2ea043; transform: translateY(-1px); }}
        button.secondary {{ background: #30363d; }}
        button.secondary:hover {{ background: #484f58; }}
        .group-list {{
            max-height: 180px;
            overflow-y: auto;
        }}
        .group-item {{
            padding: 6px 10px;
            margin: 4px 0;
            background: #21262d;
            border-radius: 4px;
            cursor: pointer;
            font-size: 0.8em;
            transition: all 0.2s;
            border-left: 3px solid transparent;
        }}
        .group-item:hover {{
            background: #30363d;
            border-left-color: #58a6ff;
        }}
        .group-item.collapsed {{
            opacity: 0.5;
            text-decoration: line-through;
        }}
        #node-details {{
            font-size: 0.75em;
            white-space: pre-wrap;
            background: #0d1117;
            padding: 10px;
            border-radius: 4px;
            max-height: 250px;
            overflow-y: auto;
            border: 1px solid #30363d;
            color: #7ee787;
        }}
        .empty-state {{ color: #484f58; font-style: italic; }}
        ::-webkit-scrollbar {{ width: 8px; }}
        ::-webkit-scrollbar-track {{ background: #0d1117; }}
        ::-webkit-scrollbar-thumb {{ background: #30363d; border-radius: 4px; }}
        ::-webkit-scrollbar-thumb:hover {{ background: #484f58; }}
    </style>
</head>
<body>
    <div id="container">
        <div id="graph-container">
            <div id="network"></div>
        </div>
        <div id="sidebar">
            <h1>🔥 TorchView Graph</h1>
            
            <div class="section">
                <h2>📊 Statistics</h2>
                <div class="stat-row">
                    <span class="stat-label">Total Nodes</span>
                    <span class="stat-value">{G.number_of_nodes()}</span>
                </div>
                <div class="stat-row">
                    <span class="stat-label">Total Edges</span>
                    <span class="stat-value">{G.number_of_edges()}</span>
                </div>
                <div class="stat-row">
                    <span class="stat-label">Tensors</span>
                    <span class="stat-value">{node_type_counts.get('tensor', 0)}</span>
                </div>
                <div class="stat-row">
                    <span class="stat-label">Functions</span>
                    <span class="stat-value">{node_type_counts.get('function', 0)}</span>
                </div>
                <div class="stat-row">
                    <span class="stat-label">Modules</span>
                    <span class="stat-value">{node_type_counts.get('module', 0)}</span>
                </div>
                <div class="stat-row">
                    <span class="stat-label">Subgraphs</span>
                    <span class="stat-value">{len(subgraphs)}</span>
                </div>
            </div>
            
            <div class="section">
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
            
            <div class="section">
                <h2>🎮 Controls</h2>
                <div class="controls">
                    <button onclick="fitNetwork()">Fit View</button>
                    <button onclick="resetLayout()" class="secondary">Reset</button>
                    <button onclick="togglePhysics()" class="secondary" id="physicsBtn">Physics: Off</button>
                </div>
            </div>
            
            <div class="section">
                <h2>📁 Module Groups</h2>
                <div class="group-list" id="group-list"></div>
            </div>
            
            <div class="section">
                <h2>📋 Node Details</h2>
                <div id="node-details" class="empty-state">Click a node to see details...</div>
            </div>
        </div>
    </div>

    <script>
        const nodesData = {json.dumps(nodes_data)};
        const edgesData = {json.dumps(edges_data)};
        const subgraphs = {json.dumps(subgraphs)};
        
        const nodes = new vis.DataSet(nodesData);
        const edges = new vis.DataSet(edgesData);
        
        const options = {{
            layout: {{
                hierarchical: {{
                    enabled: true,
                    direction: 'LR',
                    sortMethod: 'directed',
                    levelSeparation: 180,
                    nodeSpacing: 80,
                    treeSpacing: 100,
                    blockShifting: true,
                    edgeMinimization: true,
                    parentCentralization: true,
                }}
            }},
            physics: {{ enabled: false }},
            interaction: {{
                hover: true,
                tooltipDelay: 100,
                multiselect: true,
                navigationButtons: true,
                keyboard: true,
            }},
            edges: {{
                smooth: {{ type: 'cubicBezier', forceDirection: 'horizontal' }},
                width: 1.5,
            }},
            nodes: {{
                borderWidth: 2,
                shadow: {{ enabled: true, color: 'rgba(0,0,0,0.3)', x: 2, y: 2, size: 5 }},
            }},
        }};
        
        const container = document.getElementById('network');
        const network = new vis.Network(container, {{ nodes, edges }}, options);
        
        let physicsEnabled = false;
        const groupList = document.getElementById('group-list');
        const collapsedGroups = new Set();
        
        // Populate groups
        Object.entries(subgraphs).forEach(([sgId, sgInfo]) => {{
            const div = document.createElement('div');
            div.className = 'group-item';
            div.textContent = sgInfo.label;
            div.onclick = () => toggleGroup(sgId, div);
            groupList.appendChild(div);
        }});
        
        if (Object.keys(subgraphs).length === 0) {{
            groupList.innerHTML = '<div class="empty-state">No module groups</div>';
        }}
        
        network.on('click', function(params) {{
            if (params.nodes.length > 0) {{
                const nodeId = params.nodes[0];
                const node = nodes.get(nodeId);
                try {{
                    const details = JSON.parse(node.title);
                    document.getElementById('node-details').textContent = JSON.stringify(details, null, 2);
                    document.getElementById('node-details').classList.remove('empty-state');
                }} catch(e) {{
                    document.getElementById('node-details').textContent = node.title;
                }}
            }}
        }});
        
        function fitNetwork() {{ network.fit({{ animation: {{ duration: 500, easingFunction: 'easeInOutQuad' }} }}); }}
        
        function resetLayout() {{
            network.setOptions({{ layout: {{ hierarchical: {{ enabled: true }} }} }});
            setTimeout(() => {{ 
                network.setOptions({{ layout: {{ hierarchical: {{ enabled: false }} }} }});
                fitNetwork();
            }}, 100);
        }}
        
        function togglePhysics() {{
            physicsEnabled = !physicsEnabled;
            network.setOptions({{ physics: {{ enabled: physicsEnabled }} }});
            document.getElementById('physicsBtn').textContent = 'Physics: ' + (physicsEnabled ? 'On' : 'Off');
        }}
        
        function toggleGroup(groupId, element) {{
            const groupNodes = nodesData.filter(n => n.group === groupId).map(n => n.id);
            if (collapsedGroups.has(groupId)) {{
                collapsedGroups.delete(groupId);
                element.classList.remove('collapsed');
                groupNodes.forEach(nodeId => nodes.update({{ id: nodeId, hidden: false }}));
            }} else {{
                collapsedGroups.add(groupId);
                element.classList.add('collapsed');
                groupNodes.forEach(nodeId => nodes.update({{ id: nodeId, hidden: true }}));
            }}
        }}
        
        network.once('stabilizationIterationsDone', () => fitNetwork());
    </script>
</body>
</html>'''
    
    return html_content


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    create_visualizations()
