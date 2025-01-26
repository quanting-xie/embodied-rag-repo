import folium
from math import radians, sin, cos, sqrt, atan2
import json
import argparse

def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculate the distance between two points on Earth."""
    R = 6371000  # Earth's radius in meters
    
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1-a))
    return R * c

def visualize_streetview_graph(graph_file, output_file):
    """Visualize Street View graph with connections"""
    # Load graph data
    with open(graph_file, 'r') as f:
        graph_data = json.load(f)
    
    # Extract nodes and find center
    nodes = graph_data['nodes']
    
    # Debug print
    print("\nChecking connectivity data:")
    for i, node in enumerate(nodes[:5]):  # Print first 5 nodes as example
        print(f"Node {node['id']}: {len(node['connectivity'])} connections -> {node['connectivity']}")
    
    if not nodes:
        print("No nodes found in graph data")
        return
    
    # Calculate center point
    lats = [node['location']['latitude'] for node in nodes]
    lons = [node['location']['longitude'] for node in nodes]
    center_lat = sum(lats) / len(lats)
    center_lon = sum(lons) / len(lons)
    
    # Create map
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=17,
        tiles='https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}',
        attr='Google Satellite'
    )
    
    # Add points and connections
    node_locations = {}  # Store locations for connections
    
    # First, add all points
    for node in nodes:
        lat = node['location']['latitude']
        lon = node['location']['longitude']
        node_locations[node['id']] = (lat, lon)
        
        # Create popup text
        popup_text = f"""
        ID: {node['id']}<br>
        Pano ID: {node['pano_id']}<br>
        Date: {node['date']}<br>
        Connections: {len(node['connectivity'])}
        """
        
        # Add marker
        folium.CircleMarker(
            location=[lat, lon],
            radius=3,
            color='blue',
            fill=True,
            popup=popup_text
        ).add_to(m)
    
    # Then add all connections
    total_connections = 0
    for node in nodes:
        start_lat, start_lon = node_locations[node['id']]
        
        for connected_id in node['connectivity']:
            if connected_id in node_locations:  # Check if connected node exists
                end_lat, end_lon = node_locations[connected_id]
                
                # Calculate distance
                distance = haversine_distance(start_lat, start_lon, end_lat, end_lon)
                
                # Draw connection line
                folium.PolyLine(
                    locations=[[start_lat, start_lon], [end_lat, end_lon]],
                    weight=2,
                    color='green',
                    opacity=0.7,
                    popup=f'Distance: {distance:.1f}m'
                ).add_to(m)
                total_connections += 1
    
    # Add statistics to the map
    stats_html = f"""
    <div style="position: fixed; 
                bottom: 50px; 
                left: 50px; 
                width: 200px;
                height: auto;
                background-color: white;
                padding: 10px;
                border: 2px solid gray;
                border-radius: 5px;
                z-index: 1000;">
        <h4>Graph Statistics</h4>
        <p>
        Total Nodes: {len(nodes)}<br>
        Total Connections: {total_connections}<br>
        Avg. Connections: {total_connections/len(nodes):.1f}
        </p>
    </div>
    """
    m.get_root().html.add_child(folium.Element(stats_html))
    
    # Save map
    m.save(output_file)
    print(f"Visualization saved to {output_file}")
    print(f"Statistics:")
    print(f"- Total nodes: {len(nodes)}")
    print(f"- Total connections: {total_connections}")
    print(f"- Average connections per node: {total_connections/len(nodes):.1f}")

def main():
    parser = argparse.ArgumentParser(description='Visualize Street View Graph')
    parser.add_argument('--graph', type=str, required=True,
                      help='Path to streetview_graph.json file')
    parser.add_argument('--output', type=str, default='graph_visualization.html',
                      help='Output HTML file path (default: graph_visualization.html)')
    
    args = parser.parse_args()
    visualize_streetview_graph(args.graph, args.output)

if __name__ == "__main__":
    main()
