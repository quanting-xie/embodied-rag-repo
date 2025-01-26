import networkx as nx
from clustering_semantic import SemanticClusterer
from llm import LLMInterface
import logging
import asyncio
import os
from tqdm import tqdm
import numpy as np
import argparse
from datetime import datetime
from pathlib import Path
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.models import Distance
from qdrant_client.http.models import Filter, FieldCondition, MatchValue
import uuid  # Add this import at the top
import json
from config import Config
from openai import AsyncOpenAI
import sys
import pickle


# Define paths relative to project root
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
SEMANTIC_GRAPHS_DIR = os.path.join(PROJECT_ROOT, "semantic_graphs")

class SemanticGraphBuilder:
    """Simple utility class for loading and processing semantic graphs"""
    def __init__(self, relationship_extractor=None):
        self.G = nx.Graph()
        self.relationship_extractor = relationship_extractor
        self.llm = None

    def load_graph(self, filename):
        """Load graph from GML file"""
        self.G = nx.read_gml(filename)
        logging.info(f"Loaded graph with {len(self.G.nodes())} nodes")

    def get_captions(self):
        """Get all data points from the graph"""
        objects = []
        for node, data in self.G.nodes(data=True):
            position = self._parse_position(data.get('position', {}))
            obj = {
                'id': node,
                'position': position,
                'name': data.get('name', ''),
                'caption': data.get('caption', ''),
                'timestamp': data.get('timestamp', ''),
                'image_path': data.get('image_path', ''),
                'type': data.get('type', 'base')
            }
            objects.append(obj)
        return objects

    def _parse_position(self, position):
        """Parse position data with validation"""
        if isinstance(position, str):
            try:
                position = eval(position)
            except Exception:
                position = {}
        return {
            'x': float(position.get('x', 0.0)),
            'y': float(position.get('y', 0.0)),
            'z': float(position.get('z', 0.0))
        }

    async def summarize_clusters(self, G, level):
        """Process summaries for all clusters at a given level"""
        print(f"\nGenerating summaries for level {level} clusters...")
        
        # Get all clusters at this level
        clusters = [(node_id, data) for node_id, data in G.nodes(data=True) 
                   if data.get('type') == 'cluster' and data.get('level') == level]
        
        for node_id, data in clusters:
            # Get cluster members
            members = data.get('members', [])
            if not members:
                continue
                
            # Generate summary using LLM interface
            summary_data = await self.llm.generate_cluster_summary(
                "\n".join([G.nodes[m].get('caption', '') for m in members])
            )
            
            # Update cluster node with summary data
            G.nodes[node_id].update(summary_data)

    async def process_and_store_nodes(self, objects, collection_name, vector_db):
        """Process nodes and store in vector database with batch operations"""
        print("\n=== Building Semantic Forest ===")
        
        # Step 1: Build initial hierarchy
        print("\n1. Building Hierarchical Structure...")
        self.G = await self.relationship_extractor.extract_relationships(objects)
        
        # Get ALL nodes
        all_nodes = list(self.G.nodes(data=True))
        print(f"Found {len(all_nodes)} total nodes")
        
        # Debug print first few nodes
        print("\nSample nodes (first 3):")
        for node_id, data in all_nodes[:3]:
            print(f"\nNode {node_id}:")
            print(json.dumps(data, indent=2))
        
        print("\n2. Generating Embeddings")
        print(f"Processing {len(all_nodes)} embeddings...")
        
        # Prepare texts for batch embedding
        texts = []
        node_data_map = []  # Keep track of node data for each text
        
        for node_id, node_data in all_nodes:
            # Generate text for embedding
            if node_data.get('type') == 'cluster':
                text = f"""
                Name: {node_data.get('name', '')}
                Summary: {node_data.get('summary', '')}
                Position: {node_data.get('position', {})}
                Relationships: {', '.join(node_data.get('relationships', []))}
                """
            else:
                # Base node
                text = f"""
                Name: {node_data.get('name', '')}
                Caption: {node_data.get('caption', '')}
                Position: {node_data.get('position', {})}
                Image: {node_data.get('image_path', '')}
                """
            texts.append(text)
            node_data_map.append((node_id, node_data))
        
        # Generate embeddings in batches
        batch_size = 100
        embeddings = await self.llm.batch_generate_embeddings(texts, batch_size)
        
        # Create points in batches
        points = []
        failed_nodes = []
        
        print("\n3. Creating vector database points...")
        with tqdm(total=len(all_nodes), desc="Creating points") as pbar:
            for i, (embedding, (node_id, node_data)) in enumerate(zip(embeddings, node_data_map)):
                try:
                    # Create base payload with common fields
                    payload = {
                        "original_id": {
                            "node_id": node_id,
                            "graph_id": "current"
                        },
                        "date": "current",
                        "name": node_data.get('name', ''),
                        "caption": node_data.get('caption', ''),
                        "position": node_data.get('position', {}),
                        "type": node_data.get('type', 'base'),
                        "level": node_data.get('level', 0),
                        "text": texts[i]
                    }
                    
                    # Add image_path directly to payload for base nodes
                    if node_data.get('type') != 'cluster':
                        payload["image_path"] = node_data.get('image_path', '')
                        
                        # Debug print for first few nodes
                        if i < 3:
                            print(f"\nNode {i} payload:")
                            print(json.dumps(payload, indent=2))
                    
                    point = models.PointStruct(
                        id=str(uuid.uuid4()),
                        vector=embedding,
                        payload=payload
                    )
                    points.append(point)
                    
                    # Batch upload every 100 points
                    if len(points) >= 100:
                        try:
                            vector_db.upsert(
                                collection_name=collection_name,
                                points=points
                            )
                            pbar.write(f"✓ Uploaded batch of {len(points)} points")
                        except Exception as e:
                            pbar.write(f"Error uploading batch: {str(e)}")
                            failed_nodes.extend([(p.payload['original_id']['node_id'], "Batch upload failed") for p in points])
                        points = []
                        await asyncio.sleep(0.1)
                    
                except Exception as e:
                    failed_nodes.append((node_id, str(e)))
                    pbar.write(f"Error creating point for node {node_id}: {str(e)}")
                
                pbar.update(1)
        
        # Upload remaining points
        if points:
            try:
                vector_db.upsert(
                    collection_name=collection_name,
                    points=points
                )
                print(f"✓ Uploaded final {len(points)} points")
            except Exception as e:
                print(f"Error uploading final batch: {str(e)}")
                failed_nodes.extend([(p.payload['original_id']['node_id'], "Final batch upload failed") for p in points])
        
        total_points = vector_db.count(collection_name)
        print(f"\nIndexing complete:")
        print(f"- Total nodes in vector database: {total_points}")
        print(f"- Successfully indexed: {total_points} nodes")
        print(f"- Failed to index: {len(failed_nodes)} nodes")
        
        if failed_nodes:
            print("\nFailed nodes:")
            for node_id, error in failed_nodes[:10]:
                print(f"- Node {node_id}: {error}")
            if len(failed_nodes) > 10:
                print(f"... and {len(failed_nodes) - 10} more")
        
        # Debug print first few points before upload
        print("\nSample points to be uploaded (first 3):")
        for point in points[:3]:
            print(f"\nPoint ID: {point.id}")
            print(f"Payload: {json.dumps(point.payload, indent=2)}")
        
        return self.G

async def process_all_semantic_graphs(graph_path: Path, forest_name: str):
    """Process semantic graphs from either a directory or single file"""
    if graph_path.is_file():
        # Single file mode
        if graph_path.suffix == '.gml':
            semantic_graphs = [graph_path]
        else:
            print(f"Invalid file type: {graph_path}. Expected .gml file")
            return
    else:
        # Directory mode
        semantic_graphs = list(graph_path.glob("semantic_graph_*.gml"))
    
    if not semantic_graphs:
        print("No semantic graph files found")
        return
    
    print(f"\nFound {len(semantic_graphs)} semantic graphs to process:")
    for graph_file in semantic_graphs:
        print(f"- {graph_file.name}")
        
    # Process each semantic graph
    for graph_file in semantic_graphs:
        try:
            # Extract date from filename
            if '_' in graph_file.stem:
                date_str = graph_file.stem.split('_')[2]  # For semantic_graph_YYYY-MM-DD.gml format
            else:
                # Use current date for files without date in name
                date_str = datetime.now().strftime('%Y-%m-%d')
                
            print(f"\nProcessing semantic graph for date: {date_str}")
            
            # Setup output directory for this date
            output_dir = graph_file.parent / "semantic_forests"
            await generate_specialized_forests(graph_file, output_dir, forest_name)
            
        except Exception as e:
            print(f"Error processing {graph_file.name}: {str(e)}")
            continue

async def generate_specialized_forests(initial_graph_file: Path, output_dir: Path, forest_name: str):
    """Generate a semantic forest with comprehensive analysis"""
    print(f"\n=== Processing {initial_graph_file.name} ===")
    
    try:
        # Initialize components
        llm_interface = LLMInterface()
        graph_builder = SemanticGraphBuilder()
        relationship_extractor = SemanticClusterer(llm_interface)
        
        # Connect components
        relationship_extractor.summarizer = graph_builder
        graph_builder.relationship_extractor = relationship_extractor
        graph_builder.llm = llm_interface
        
        # Create output directories with proper structure
        base_dir = Path(PROJECT_ROOT)
        forest_dir = base_dir / "semantic_forests" / forest_name
        forest_dir.mkdir(parents=True, exist_ok=True)
        
        vector_db_dir = forest_dir / "vector_db"
        vector_db_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Output directory: {forest_dir}")
        print(f"Vector database: {vector_db_dir}")
        
        # Setup vector database
        vector_db = QdrantClient(path=str(vector_db_dir))
        collection_name = f"nodes_{forest_name}"
        
        try:
            collections = vector_db.get_collections()
            collection_exists = any(c.name == collection_name for c in collections.collections)
            
            if collection_exists:
                print(f"\nFound existing collection: {collection_name}")
                user_input = input("Do you want to recreate the collection? (y/N): ").lower()
                if user_input == 'y':
                    print(f"Recreating collection {collection_name}...")
                    # Delete existing collection
                    vector_db.delete_collection(collection_name=collection_name)
                    # Create new collection
                    vector_db.create_collection(
                        collection_name=collection_name,
                        vectors_config=models.VectorParams(
                            size=1536,
                            distance=models.Distance.COSINE
                        )
                    )
                    print(f"✓ Collection {collection_name} recreated")
                else:
                    print("Using existing collection")
            else:
                print(f"Creating new vector database collection: {collection_name}")
                vector_db.create_collection(
                    collection_name=collection_name,
                    vectors_config=models.VectorParams(
                        size=1536,
                        distance=models.Distance.COSINE
                    )
                )
            
        except Exception as e:
            print(f"Error setting up vector database: {str(e)}")
            raise
        
        # Load and process graph
        print("\nLoading initial graph...")
        graph_builder.load_graph(initial_graph_file)
        all_objects = graph_builder.get_captions()
        
        # Process and store nodes
        forest = await graph_builder.process_and_store_nodes(all_objects, collection_name, vector_db)
        
        # Save the forest with proper path
        forest_file = forest_dir / f"semantic_forest_{forest_name}.gml"
        print(f"\nSaving forest to {forest_file}")
        forest_dir.mkdir(parents=True, exist_ok=True)  # Ensure directory exists
        nx.write_gml(forest, str(forest_file))  # Convert Path to string
        print("✓ Forest saved")
        
        return forest_file
        
    except Exception as e:
        print(f"Error generating forest: {str(e)}")
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate Semantic Forests')
    parser.add_argument('--input-dir', type=str, help='Path to semantic graph file or directory')
    parser.add_argument('--name', type=str, help='Name for the semantic forest and vector database')
    args = parser.parse_args()
    
    if not args.name:
        print("Please provide a name for the semantic forest using --name")
        sys.exit(1)
    
    try:
        # Setup paths
        if args.input_dir:
            graph_path = Path(args.input_dir)
        else:
            graph_path = Path(PROJECT_ROOT) / "graph"
        
        if not graph_path.exists():
            raise ValueError(f"Path not found: {graph_path}")
            
        print(f"Processing semantic graphs from: {graph_path}")
        print(f"Using name: {args.name}")
        
        # Process semantic graphs with custom name
        asyncio.run(process_all_semantic_graphs(graph_path, args.name))
        
    except Exception as e:
        print(f"\nError during processing: {str(e)}")
        import traceback
        print(traceback.format_exc())
