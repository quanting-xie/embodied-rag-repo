import networkx as nx
import plotly.graph_objects as go
import numpy as np
import os
import colorsys

def get_distinct_colors(n):
    """Generate n distinct colors using HSV color space"""
    colors = []
    for i in range(n):
        hue = i / n
        saturation = 0.7 + np.random.random() * 0.3  # Random between 0.7-1.0
        value = 0.7 + np.random.random() * 0.3       # Random between 0.7-1.0
        rgb = colorsys.hsv_to_rgb(hue, saturation, value)
        colors.append(f'rgb({int(rgb[0]*255)},{int(rgb[1]*255)},{int(rgb[2]*255)})')
    return colors

def visualize_enhanced_graph_3d(G, output_path=None):
    """Visualize the semantic forest with interactive highlighting"""
    pos = {}
    
    # Calculate appropriate z_increment based on geographic scale
    lat_coords = []
    lon_coords = []
    for node, data in G.nodes(data=True):
        if 'position' in data:
            position = data['position']
            if isinstance(position, str):
                try:
                    position = eval(position)
                except:
                    continue
            lat_coords.append(float(position.get('y', 0.0)))
            lon_coords.append(float(position.get('x', 0.0)))
    
    # Calculate geographic ranges and z_increment
    lat_range = max(lat_coords) - min(lat_coords)
    lon_range = max(lon_coords) - min(lon_coords)
    z_increment = min(lat_range, lon_range) * 0.1
    
    # Get all clusters and their members
    clusters = {}
    for node, data in G.nodes(data=True):
        if data.get('type') == 'cluster':
            members = data.get('members', [])
            if isinstance(members, str):  # Handle single member case
                members = [members]
            clusters[node] = set(members)
            
            # Add nested clusters' members
            nested_members = set()
            for member in members:
                if member.startswith('cluster_'):
                    if G.nodes[member].get('members'):
                        nested_members.update(G.nodes[member].get('members', []))
            clusters[node].update(nested_members)
    
    # Generate distinct colors for clusters
    cluster_colors = get_distinct_colors(len(clusters))
    color_map = dict(zip(clusters.keys(), cluster_colors))
    
    # Create node to color mapping including members
    node_colors = {}
    for cluster, members in clusters.items():
        cluster_color = color_map[cluster]
        node_colors[cluster] = cluster_color
        for member in members:
            node_colors[member] = cluster_color
    
    # First pass: Add all nodes with positions
    for node, data in G.nodes(data=True):
        if 'position' in data:
            level = data.get('level', 0)
            position = data['position']
            if isinstance(position, str):
                try:
                    position = eval(position)
                except:
                    print(f"Warning: Could not parse position for node {node}")
                    position = {'x': 0.0, 'y': 0.0, 'z': 0.0}
            
            pos[node] = [
                float(position.get('x', 0.0)),
                float(position.get('y', 0.0)),
                level * z_increment
            ]
    
    # Create figure
    fig = go.Figure()
    
    # Keep track of nodes we've already added
    added_nodes = set()
    
    # Store node hierarchy relationships
    node_hierarchy = {}  # Store parent-child relationships
    for cluster, members in clusters.items():
        node_hierarchy[cluster] = {
            'parent': None,
            'children': set(members),
            'all_connected': set(members)  # Will include all hierarchically connected nodes
        }
        for member in members:
            node_hierarchy[member] = {
                'parent': cluster,
                'children': set(),
                'all_connected': {cluster}
            }
    
    # Propagate connections up and down the hierarchy
    for node in node_hierarchy:
        current = node
        # Go up the hierarchy
        while node_hierarchy[current]['parent']:
            parent = node_hierarchy[current]['parent']
            node_hierarchy[node]['all_connected'].add(parent)
            node_hierarchy[parent]['all_connected'].add(node)
            current = parent
    
    # Process each cluster with updated trace naming and visibility
    for cluster_idx, (cluster, members) in enumerate(clusters.items()):
        cluster_color = color_map[cluster]
        cluster_name = G.nodes[cluster].get('name', cluster)
        
        # Create traces for this cluster and its members
        edge_x, edge_y, edge_z = [], [], []
        node_x, node_y, node_z = [], [], []
        node_text = []
        node_size = []
        node_ids = []  # Store node IDs for reference
        
        # Add cluster node and its edges
        if cluster in pos and cluster not in added_nodes:
            x, y, z = pos[cluster]
            node_x.append(x)
            node_y.append(y)
            node_z.append(z)
            node_text.append(cluster_name)
            node_size.append(15)
            node_ids.append(cluster)
            added_nodes.add(cluster)
            
            # Add edges
            for member in G.nodes[cluster].get('members', []):
                if member in pos:
                    x1, y1, z1 = pos[member]
                    edge_x.extend([x, x1, None])
                    edge_y.extend([y, y1, None])
                    edge_z.extend([z, z1, None])
        
        # Add member nodes
        for member in members:
            if member in pos and member not in added_nodes:
                x, y, z = pos[member]
                node_x.append(x)
                node_y.append(y)
                node_z.append(z)
                member_name = G.nodes[member].get('name', member)
                node_text.append(member_name)
                node_size.append(8)
                node_ids.append(member)
                added_nodes.add(member)
        
        # Add edges trace with custom data
        if edge_x:
            fig.add_trace(go.Scatter3d(
                x=edge_x, y=edge_y, z=edge_z,
                mode='lines',
                line=dict(color=cluster_color, width=2),
                opacity=0.5,
                hoverinfo='none',
                showlegend=False,
                customdata=[cluster] * len(edge_x),  # Store cluster ID
                name=f"edges_{cluster}"  # Unique name for edges
            ))
        
        # Add nodes trace with custom data
        if node_x:
            fig.add_trace(go.Scatter3d(
                x=node_x, y=node_y, z=node_z,
                mode='markers',
                marker=dict(
                    size=node_size,
                    color=cluster_color,
                    line=dict(color=cluster_color, width=0.5)
                ),
                text=node_text,
                hoverinfo='text',
                name=cluster_name,  # Use actual cluster name
                customdata=node_ids,  # Store node IDs
                showlegend=True
            ))
    
    # Add click event handling
    fig.update_layout(
        clickmode='event+select',
        # ... (previous layout settings) ...
    )
    
    # Add updatemenus for reset button
    fig.update_layout(
        updatemenus=[
            dict(
                type="buttons",
                showactive=False,
                buttons=[
                    dict(
                        label="Reset View",
                        method="update",
                        args=[{"opacity": 1.0}],  # Reset all opacities
                    )
                ],
                x=0.9,
                y=1.1,
            )
        ]
    )
    
    # Add click handling JavaScript
    fig.update_layout(
        newshape_line_color='cyan',
        clickmode='event+select',
    )
    
    # Add JavaScript for interactivity
    fig.add_annotation(
        text="Click on legend items to highlight connected nodes",
        xref="paper", yref="paper",
        x=0, y=1.1,
        showarrow=False,
    )
    
    # Add JavaScript for click handling
    js_code = """
    <script>
        var graphDiv = document.getElementById('graph');
        graphDiv.on('plotly_click', function(data) {
            var point = data.points[0];
            var clickedNode = point.customdata;
            var traces = graphDiv.data;
            
            // Reset all opacities first
            traces.forEach(function(trace) {
                Plotly.restyle(graphDiv, {'opacity': 0.2}, [trace.index]);
            });
            
            // Highlight connected nodes and edges
            traces.forEach(function(trace) {
                if (trace.customdata && trace.customdata.includes(clickedNode)) {
                    Plotly.restyle(graphDiv, {'opacity': 1.0}, [trace.index]);
                }
            });
        });
    </script>
    """
    
    if output_path:
        if not output_path.endswith('.html'):
            output_path += '.html'
        os.makedirs('plots', exist_ok=True)
        
        # Write HTML with added JavaScript
        with open(os.path.join('plots', output_path), 'w') as f:
            html_str = fig.to_html(include_plotlyjs=True, full_html=True)
            html_str = html_str.replace('</body>', f'{js_code}</body>')
            f.write(html_str)
            
        print(f"Saved interactive visualization to: plots/{output_path}")
    
    fig.show()

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python graph_visualizer.py <path_to_semantic_forest.gml> [output_path]")
        sys.exit(1)
    
    graph_file = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else None
    
    G = nx.read_gml(graph_file)
    visualize_enhanced_graph_3d(G, output_path)
