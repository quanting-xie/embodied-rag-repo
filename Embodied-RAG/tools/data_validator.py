import json
import networkx as nx
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import random
from pathlib import Path
import os
import pprint

class DataValidatorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Dataset Validator")
        
        # Load data
        self.metadata_path = "/Users/danielxie/Embodied-RAG_datasets/tokyo/metadata/streetview_graph.json"
        self.graph_path = "/Users/danielxie/E-RAG/Embodied-RAG/graph/tokyo_streetview_graph.gml"
        self.images_dir = "/Users/danielxie/Embodied-RAG_datasets/tokyo/images/merged"
        
        self.load_data()
        
        # Setup GUI
        self.setup_gui()
        
        # Load first sample
        self.current_index = 0
        self.load_next_sample()

    def load_data(self):
        print("Loading data...")
        # Load metadata
        with open(self.metadata_path, 'r') as f:
            data = json.load(f)
            
        self.metadata_info = data['metadata']
        self.nodes = data['nodes']
            
        # Debug print metadata structure
        print("\nMetadata info:", self.metadata_info)
        print("\nFirst node example:")
        if self.nodes:
            print(json.dumps(self.nodes[0], indent=2))
        
        # Load graph
        self.graph = nx.read_gml(self.graph_path)
        print(f"\nGraph nodes: {len(self.graph.nodes)}")
        
        # Create sample list from nodes
        self.samples = list(range(len(self.nodes)))
        random.shuffle(self.samples)
        print(f"Loaded {len(self.samples)} samples")

    def setup_gui(self):
        # Main frame
        self.main_frame = ttk.Frame(self.root, padding="10")
        self.main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Left frame for image
        self.image_frame = ttk.Frame(self.main_frame)
        self.image_frame.grid(row=0, column=0, padx=5, pady=5)
        
        # Image label
        self.image_label = ttk.Label(self.image_frame)
        self.image_label.grid(row=0, column=0)
        
        # Right frame for info
        self.info_frame = ttk.Frame(self.main_frame)
        self.info_frame.grid(row=0, column=1, padx=5, pady=5, sticky=(tk.N, tk.S))
        
        # Sample ID
        ttk.Label(self.info_frame, text="Sample ID:").grid(row=0, column=0, sticky=tk.W)
        self.id_text = tk.Text(self.info_frame, wrap=tk.WORD, width=50, height=1)
        self.id_text.grid(row=1, column=0, pady=5)
        
        # Location
        ttk.Label(self.info_frame, text="Location:").grid(row=2, column=0, sticky=tk.W)
        self.location_text = tk.Text(self.info_frame, wrap=tk.WORD, width=50, height=3)
        self.location_text.grid(row=3, column=0, pady=5)
        
        # Caption
        ttk.Label(self.info_frame, text="Caption:").grid(row=4, column=0, sticky=tk.W)
        self.caption_text = tk.Text(self.info_frame, wrap=tk.WORD, width=50, height=10)
        self.caption_text.grid(row=5, column=0, pady=5)
        
        # Node Metadata
        ttk.Label(self.info_frame, text="Node Metadata:").grid(row=6, column=0, sticky=tk.W)
        self.metadata_text = tk.Text(self.info_frame, wrap=tk.WORD, width=50, height=15)
        self.metadata_text.grid(row=7, column=0, pady=5)
        
        # Graph Data
        ttk.Label(self.info_frame, text="Graph Data:").grid(row=8, column=0, sticky=tk.W)
        self.graph_text = tk.Text(self.info_frame, wrap=tk.WORD, width=50, height=15)
        self.graph_text.grid(row=9, column=0, pady=5)
        
        # Navigation buttons
        self.button_frame = ttk.Frame(self.main_frame)
        self.button_frame.grid(row=1, column=0, columnspan=2, pady=10)
        
        ttk.Button(self.button_frame, text="Previous", command=self.prev_sample).grid(row=0, column=0, padx=5)
        ttk.Button(self.button_frame, text="Next", command=self.next_sample).grid(row=0, column=1, padx=5)
        ttk.Button(self.button_frame, text="Random", command=self.random_sample).grid(row=0, column=2, padx=5)

    def load_next_sample(self):
        # Get current sample index and data
        sample_idx = self.samples[self.current_index]
        
        # Get graph node data
        graph_node = self.nodes[sample_idx]
        node_id = graph_node.get('id')
        image_id = f"streetview_{node_id:04d}"
        
        # Load and display image
        img_path = os.path.join(self.images_dir, f"{image_id}.jpg")
        print(f"Looking for image at: {img_path}")
        
        if os.path.exists(img_path):
            try:
                img = Image.open(img_path)
                # Resize image while maintaining aspect ratio
                img.thumbnail((800, 800))
                photo = ImageTk.PhotoImage(img)
                self.image_label.configure(image=photo)
                self.image_label.image = photo
            except Exception as e:
                self.image_label.configure(text=f"Error loading image: {str(e)}")
        else:
            self.image_label.configure(text=f"Image not found: {img_path}")
        
        # Display ID
        self.id_text.delete('1.0', tk.END)
        self.id_text.insert('1.0', f"ID: {node_id}")
        
        # Get graph data for caption and location
        graph_data = None
        if str(node_id) in self.graph:
            graph_data = self.graph.nodes[str(node_id)]
        
        # Display caption from graph data
        self.caption_text.delete('1.0', tk.END)
        if graph_data and 'caption' in graph_data:
            self.caption_text.insert('1.0', graph_data['caption'])
        else:
            self.caption_text.insert('1.0', 'No caption available')
        
        # Display location
        self.location_text.delete('1.0', tk.END)
        if 'location' in graph_node:
            location = graph_node['location']
            location_str = f"Latitude: {location['latitude']}\nLongitude: {location['longitude']}"
        else:
            location_str = "Location not available"
        self.location_text.insert('1.0', location_str)
        
        # Format and display node metadata
        metadata_str = "Node Metadata:\n"
        metadata_str += json.dumps(graph_node, indent=2)
        self.metadata_text.delete('1.0', tk.END)
        self.metadata_text.insert('1.0', metadata_str)
        
        # Display graph data
        self.graph_text.delete('1.0', tk.END)
        if graph_data:
            graph_str = "Graph Data:\n" + json.dumps(graph_data, indent=2)
        else:
            graph_str = f"Node not found in graph (ID: {node_id})"
        self.graph_text.insert('1.0', graph_str)

    def next_sample(self):
        self.current_index = (self.current_index + 1) % len(self.samples)
        self.load_next_sample()

    def prev_sample(self):
        self.current_index = (self.current_index - 1) % len(self.samples)
        self.load_next_sample()

    def random_sample(self):
        self.current_index = random.randint(0, len(self.samples) - 1)
        self.load_next_sample()

def main():
    root = tk.Tk()
    app = DataValidatorGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main() 