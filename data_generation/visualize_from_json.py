import json
import folium
from pathlib import Path

def visualize_points_from_json(json_path, save_path=None):
    """
    Visualize points from streetview_graph.json
    
    Args:
        json_path: Path to the streetview_graph.json file
        save_path: Path to save the HTML visualization
    """
    # Load JSON data
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Extract points from nodes
    points = []
    for node in data.get('nodes', []):
        try:
            # Get coordinates directly from node
            lat = float(node.get('latitude'))
            lon = float(node.get('longitude'))
            pano_id = node.get('pano_id')
            heading = node.get('heading')
            date = node.get('date')
            image_path = node.get('image', {}).get('path')
            
            points.append({
                'lat': lat,
                'lon': lon,
                'id': node.get('id'),
                'pano_id': pano_id,
                'heading': heading,
                'date': date,
                'image_path': image_path
            })
        except (TypeError, ValueError) as e:
            print(f"Error processing node {node.get('id', 'unknown')}: {e}")
            continue
    
    if not points:
        print("No valid points found in JSON file")
        return
    
    # Calculate center point (average of all points)
    center_lat = sum(p['lat'] for p in points) / len(points)
    center_lon = sum(p['lon'] for p in points) / len(points)
    
    # Create map
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=17,
        tiles='https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}',
        attr='Google Satellite'
    )
    
    # Add points to map
    for point in points:
        # Create popup content with detailed information
        popup_content = f"""
        <b>ID:</b> {point['id']}<br>
        <b>Pano ID:</b> {point['pano_id']}<br>
        <b>Heading:</b> {point['heading']}<br>
        <b>Date:</b> {point['date']}<br>
        <b>Image:</b> {point['image_path']}<br>
        <b>Coordinates:</b> {point['lat']}, {point['lon']}
        """
        
        folium.CircleMarker(
            location=[point['lat'], point['lon']],
            radius=3,
            color='blue',
            fill=True,
            popup=folium.Popup(popup_content, max_width=300)
        ).add_to(m)
    
    # Add connections between nearby points
    max_connection_distance = 15  # meters
    for i, point1 in enumerate(points):
        for point2 in points[i+1:]:
            # Calculate rough distance
            dist_lat = abs(point1['lat'] - point2['lat'])
            dist_lon = abs(point1['lon'] - point2['lon'])
            if dist_lat < 0.0001 and dist_lon < 0.0001:  # rough approximation
                folium.PolyLine(
                    locations=[[point1['lat'], point1['lon']], 
                             [point2['lat'], point2['lon']]],
                    weight=2,
                    color='red',
                    opacity=0.5
                ).add_to(m)
    
    # Add a legend
    legend_html = '''
    <div style="position: fixed; 
                bottom: 50px; right: 50px; width: 150px; height: 90px; 
                border:2px solid grey; z-index:9999; 
                background-color:white;
                padding: 10px;
                font-size: 14px;">
        <b>Legend</b><br>
        <i class="fa fa-circle" style="color:blue"></i> Panorama Location<br>
        <i class="fa fa-minus" style="color:red"></i> Connection
    </div>
    '''
    m.get_root().html.add_child(folium.Element(legend_html))
    
    # Save map
    if save_path:
        m.save(save_path)
        print(f"Map saved to {save_path}")
    
    return m

def main():
    # Get dataset name from user
    dataset_name = input("Enter the dataset name (e.g., CMU_500_front): ").strip()
    if not dataset_name:
        print("Dataset name cannot be empty")
        return
    
    # Construct paths
    base_dir = Path('datasets') / dataset_name
    json_path = base_dir / 'metadata' / 'streetview_graph.json'
    vis_dir = base_dir / 'visualization'
    vis_dir.mkdir(parents=True, exist_ok=True)
    save_path = vis_dir / 'points_visualization.html'
    
    # Check if JSON file exists
    if not json_path.exists():
        print(f"JSON file not found: {json_path}")
        return
    
    print(f"Loading data from: {json_path}")
    visualize_points_from_json(json_path, save_path)

if __name__ == "__main__":
    main()
