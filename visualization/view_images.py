import json
import argparse
from PIL import Image
import matplotlib.pyplot as plt
import os

def load_graph(graph_file):
    """Load the graph data from JSON"""
    with open(graph_file, 'r') as f:
        return json.load(f)

def find_node(graph_data, node_id=None, pano_id=None, lat=None, lon=None):
    """Find a node by ID, pano_id, or closest to coordinates"""
    if node_id is not None:
        for node in graph_data['nodes']:
            if node['id'] == node_id:
                return node
    
    if pano_id is not None:
        for node in graph_data['nodes']:
            if node['pano_id'] == pano_id:
                return node
    
    if lat is not None and lon is not None:
        closest_node = None
        min_distance = float('inf')
        for node in graph_data['nodes']:
            dlat = abs(node['location']['latitude'] - lat)
            dlon = abs(node['location']['longitude'] - lon)
            distance = (dlat**2 + dlon**2)**0.5
            if distance < min_distance:
                min_distance = distance
                closest_node = node
        return closest_node
    
    return None

def display_node_images(node, show_separate=False):
    """Display images for a given node"""
    if show_separate:
        # Display separate images
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle(f"Node {node['id']} - Pano {node['pano_id']} - Date: {node['date']}")
        
        for i, img_info in enumerate(node['images']['separate']):
            img_path = img_info['path']
            if os.path.exists(img_path):
                img = Image.open(img_path)
                axes[i].imshow(img)
                axes[i].set_title(f"Heading: {img_info['heading']}°")
                axes[i].axis('off')
            else:
                print(f"Image not found: {img_path}")
    else:
        # Display merged image
        plt.figure(figsize=(15, 5))
        img_path = node['images']['merged']['path']
        if os.path.exists(img_path):
            img = Image.open(img_path)
            plt.imshow(img)
            plt.title(f"Node {node['id']} - Pano {node['pano_id']} - Date: {node['date']}")
            plt.axis('off')
        else:
            print(f"Image not found: {img_path}")
    
    plt.show()

def display_node_info(node):
    """Display information about the node"""
    print("\nNode Information:")
    print(f"ID: {node['id']}")
    print(f"Panorama ID: {node['pano_id']}")
    print(f"Date: {node['date']}")
    print(f"Location: {node['location']['latitude']}, {node['location']['longitude']}")
    
    # Print image paths
    print("\nImage paths:")
    if 'images' in node:
        if 'merged' in node['images']:
            print(f"Merged: {node['images']['merged']['path']}")
        if 'separate' in node['images']:
            print("\nSeparate images:")
            for img in node['images']['separate']:
                print(f"- {img['path']} (heading: {img['heading']}°)")

def main():
    parser = argparse.ArgumentParser(description='View images from Street View graph')
    parser.add_argument('--graph', type=str, required=True,
                      help='Path to streetview_graph.json file')
    parser.add_argument('--node-id', type=int,
                      help='Node ID to display')
    parser.add_argument('--pano-id', type=str,
                      help='Panorama ID to display')
    parser.add_argument('--lat', type=float,
                      help='Latitude to find closest node')
    parser.add_argument('--lon', type=float,
                      help='Longitude to find closest node')
    parser.add_argument('--separate', action='store_true',
                      help='Show separate images instead of merged')
    
    args = parser.parse_args()
    
    # Load graph data
    graph_data = load_graph(args.graph)
    
    # Find requested node
    node = find_node(graph_data, args.node_id, args.pano_id, args.lat, args.lon)
    
    if node:
        display_node_info(node)
        display_node_images(node, args.separate)
    else:
        print("Node not found!")
        if args.node_id:
            print(f"No node with ID: {args.node_id}")
        if args.pano_id:
            print(f"No node with panorama ID: {args.pano_id}")
        if args.lat and args.lon:
            print(f"Could not find node near coordinates: {args.lat}, {args.lon}")

if __name__ == "__main__":
    main()
