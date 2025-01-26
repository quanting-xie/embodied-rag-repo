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
        saturation = 0.7 + np.random.random() * 0.3
        value = 0.7 + np.random.random() * 0.3
        rgb = colorsys.hsv_to_rgb(hue, saturation, value)
        colors.append(f'rgb({int(rgb[0]*255)},{int(rgb[1]*255)},{int(rgb[2]*255)})')
    return colors

def get_position_dict(pos_data):
    """Helper function to handle position data in both string and dict formats"""
    if isinstance(pos_data, str):
        try:
            return eval(pos_data)
        except:
            return {'x': 0.0, 'y': 0.0}
    elif isinstance(pos_data, dict):
        return pos_data
    return {'x': 0.0, 'y': 0.0}

def visualize_graph_mapbox(G, mapbox_token, output_path=None):
    """Visualize the semantic forest overlaid on a 3D map"""
    # Get all clusters and their members
    clusters = {}
    for node, data in G.nodes(data=True):
        if data.get('type') == 'cluster':
            members = data.get('members', [])
            if isinstance(members, str):
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
    
    # Create figure
    fig = go.Figure()
    
    # Keep track of nodes we've already added
    added_nodes = set()
    
    # Process each cluster
    for cluster_idx, (cluster, members) in enumerate(clusters.items()):
        cluster_color = color_map[cluster]
        
        # Create lists for node positions
        lats, lons = [], []
        node_text = []
        node_size = []
        
        # Add cluster node
        if cluster in G.nodes and 'position' in G.nodes[cluster] and cluster not in added_nodes:
            pos = G.nodes[cluster]['position']
            if isinstance(pos, str):
                pos = eval(pos)
            lats.append(float(pos.get('y', 0)))
            lons.append(float(pos.get('x', 0)))
            node_text.append(G.nodes[cluster].get('name', cluster))
            node_size.append(15)
            added_nodes.add(cluster)
        
        # Add member nodes
        for member in members:
            if member in G.nodes and 'position' in G.nodes[member] and member not in added_nodes:
                pos = G.nodes[member]['position']
                if isinstance(pos, str):
                    pos = eval(pos)
                lats.append(float(pos.get('y', 0)))
                lons.append(float(pos.get('x', 0)))
                node_text.append(G.nodes[member].get('name', member))
                node_size.append(8)
                added_nodes.add(member)
        
        # Add nodes for this cluster
        if lats:
            fig.add_trace(go.Scattermapbox(
                lat=lats,
                lon=lons,
                mode='markers',
                marker=dict(
                    size=node_size,
                    color=cluster_color,
                    opacity=0.7
                ),
                text=node_text,
                hoverinfo='text',
                name=f"Cluster {cluster_idx + 1}",
                showlegend=True
            ))
            
            # Add lines between cluster and its immediate members
            if cluster in G.nodes:
                cluster_pos = G.nodes[cluster]['position']
                if isinstance(cluster_pos, str):
                    cluster_pos = eval(cluster_pos)
                cluster_lat = float(cluster_pos.get('y', 0))
                cluster_lon = float(cluster_pos.get('x', 0))
                
                immediate_members = G.nodes[cluster].get('members', [])
                if isinstance(immediate_members, str):
                    immediate_members = [immediate_members]
                
                for member in immediate_members:
                    if member in G.nodes and 'position' in G.nodes[member]:
                        member_pos = G.nodes[member]['position']
                        if isinstance(member_pos, str):
                            member_pos = eval(member_pos)
                        member_lat = float(member_pos.get('y', 0))
                        member_lon = float(member_pos.get('x', 0))
                        
                        fig.add_trace(go.Scattermapbox(
                            lat=[cluster_lat, member_lat],
                            lon=[cluster_lon, member_lon],
                            mode='lines',
                            line=dict(
                                color=cluster_color,
                                width=2
                            ),
                            opacity=0.5,
                            showlegend=False
                        ))
    
    # Calculate center coordinates safely
    positions = []
    for n in G.nodes:
        if 'position' in G.nodes[n]:
            pos = get_position_dict(G.nodes[n]['position'])
            positions.append((float(pos.get('y', 0)), float(pos.get('x', 0))))
    
    center_lat = np.mean([p[0] for p in positions])
    center_lon = np.mean([p[1] for p in positions])

    # Update layout with mapbox configuration
    fig.update_layout(
        title='Semantic Forest on Map',
        mapbox=dict(
            style='satellite-streets',
            accesstoken=mapbox_token,
            center=dict(
                lat=center_lat,
                lon=center_lon
            ),
            zoom=15,
            pitch=45
        ),
        margin=dict(l=0, r=0, t=30, b=0),
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        )
    )
    
    if output_path:
        if not output_path.endswith('.html'):
            output_path += '.html'
        os.makedirs('plots', exist_ok=True)
        fig.write_html(os.path.join('plots', output_path))
        print(f"Saved visualization to: plots/{output_path}")
    
    fig.show()

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python graph_visualizer_mapbox.py <path_to_semantic_forest.gml> [output_path]")
        sys.exit(1)
    
    graph_file = sys.argv[1]
    mapbox_token = "pk.eyJ1IjoieGllcXVhbnRpbmciLCJhIjoiY201c21ka3VpMG52aDJqcHJwZXE4YTljaCJ9.KWo6cIwM005U-5l4Q5lI2w"
    output_path = sys.argv[2] if len(sys.argv) > 2 else None
    
    G = nx.read_gml(graph_file)
    visualize_graph_mapbox(G, mapbox_token, output_path) 