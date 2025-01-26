import networkx as nx
from llm import LLMInterface
import asyncio
import os
from typing import Dict, List, Optional, Union
import json
from datetime import datetime
import re
import traceback
from pathlib import Path
from qdrant_client import QdrantClient
from qdrant_client.http import models
import numpy as np
from openai import AsyncOpenAI
import uuid
import config
from tqdm import tqdm

class EnvironmentalChat:
    def __init__(self):
        print("Initializing Environmental Analysis Chat...")
        self.llm = LLMInterface()
        self.forests = {}
        self.chat_history = []
        self.context_window_size = config.Config.RETRIEVAL['context_window_size']
        
        # Update paths using config
        self.forest_path = config.Config.PATHS['graph_path']
        
        # Initialize vector database
        self.vector_db_path = Path(config.Config.PATHS['vector_db_path'])
        self.vector_db_path.mkdir(exist_ok=True)
        print(f"Using vector database at: {self.vector_db_path}")
        
        # Extract collection name from graph path
        graph_name = Path(self.forest_path).stem.replace('semantic_forest_', '')
        self.collection_name = f"nodes_{graph_name}"
        
        self.vector_db = QdrantClient(path=str(self.vector_db_path))
        print(f"Using collection name: {self.collection_name}")
        self.setup_vector_db()
        
        self.available_dates = []

    @classmethod
    async def create(cls):
        """Async factory method to properly initialize the class"""
        self = cls()
        # Load all available forests and index their nodes
        self.available_dates = await self.load_all_forests()
        print("Initialization complete.")
        return self

    def setup_vector_db(self):
        """Setup vector database collection if it doesn't exist"""
        try:
            # Check if collection exists
            collections = self.vector_db.get_collections()
            collection_exists = any(c.name == self.collection_name for c in collections.collections)
            
            print(f"\nVector DB Setup:")
            print(f"Collection name: {self.collection_name}")
            print(f"Available collections: {[c.name for c in collections.collections]}")
            
            if not collection_exists:
                print(f"Creating new vector database collection: {self.collection_name}")
                self.vector_db.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=models.VectorParams(
                        size=config.Config.VECTOR_DB['embedding_dim'],
                        distance=models.Distance[config.Config.VECTOR_DB['distance_metric']]
                    )
                )
            else:
                # Get collection info
                collection_info = self.vector_db.get_collection(self.collection_name)
                print(f"Using existing vector database collection: {self.collection_name}")
                print(f"Collection contains {collection_info.points_count:,} points")
                
                # List some sample points
                sample_points = self.vector_db.scroll(
                    collection_name=self.collection_name,
                    limit=2,
                    with_payload=True
                )[0]
                if sample_points:
                    print("\nSample points:")
                    for point in sample_points:
                        print(f"ID: {point.id}")
                        print(f"Payload: {point.payload}")
            
        except Exception as e:
            print(f"Error setting up vector database: {str(e)}")
            print(traceback.format_exc())

    async def load_all_forests(self) -> List[str]:
        """Load all forests from semantic forests directory"""
        print(f"Loading forests from: {self.forest_path}")
        available_dates = []
        
        # Convert string path to Path object
        graph_file = Path(self.forest_path)
        
        if graph_file.exists():
            try:
                # Load the graph
                G = nx.read_gml(graph_file)
                date_str = "current"  # Use a single date identifier
                self.forests[date_str] = G
                available_dates.append(date_str)
                print(f"✓ Loaded forest from {graph_file}")
            except Exception as e:
                print(f"✗ Error loading forest: {str(e)}")
                print(traceback.format_exc())
        else:
            print(f"Error: Graph file not found at {graph_file}")
        
        if not available_dates:
            print("No valid forests found!")
        else:
            print(f"\nSuccessfully loaded forest")
            print(f"Total nodes in vector database: {self.vector_db.count(self.collection_name)}")
        
        return available_dates

    async def retrieve_hierarchical_context(self, query: str, agent_location: Dict = None, max_chars: int = None, return_nodes: bool = False) -> Union[str, List[Dict]]:
        """
        Retrieve context by finding relevant base nodes and their parent hierarchy.
        If return_nodes is True, returns the reranked base nodes instead of context string.
        """
        try:
            # Generate query embedding
            query_embedding = await self.llm.client.embeddings.create(
                model="text-embedding-ada-002",
                input=query
            )
 
            # Now try with corrected level filter
            must_conditions = [
                models.FieldCondition(
                    key="level",  # Changed from node_data.level to level
                    match=models.MatchValue(value=0)
                )
            ]
            
            print(f"Filter conditions: {must_conditions}")
            
            base_nodes = self.vector_db.search(
                collection_name=self.collection_name,
                query_vector=query_embedding.data[0].embedding,
                query_filter=models.Filter(must=must_conditions),
                limit=config.Config.RETRIEVAL['search_params']['limit'],
                with_payload=True,
                score_threshold=0.0  # Remove threshold temporarily for debugging
            )
            
            print(f"\nDebug: Found {len(base_nodes)} base nodes with filter")

            # Print initial semantic scores
            print("\nTop-k semantic scores before spatial adjustment:")
            for i, node in enumerate(base_nodes[:10]):
                print(f"{i+1}. {node.payload.get('name', 'unnamed')}: {node.score:.4f}")
            
            # Rerank with spatial information if available
            reranked_nodes = []
            semantic_weight = config.Config.RETRIEVAL['reranking']['semantic_weight']
            spatial_weight = config.Config.RETRIEVAL['reranking']['spatial_weight']
            
            for node in base_nodes:
                try:
                    # Get position directly from payload
                    position = node.payload.get('position', {})
                    
                    if agent_location:
                        # Calculate spatial score
                        lat1 = float(position.get('y', 0))
                        lon1 = float(position.get('x', 0))
                        lat2 = float(agent_location['latitude'])
                        lon2 = float(agent_location['longitude'])
                        
                        # Calculate distance
                        R = 6371000  # Earth's radius in meters
                        phi1, phi2 = np.radians(lat1), np.radians(lat2)
                        dphi = np.radians(lat2 - lat1)
                        dlambda = np.radians(lon2 - lon1)
                        a = np.sin(dphi/2)**2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda/2)**2
                        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
                        dist = R * c
                        
                        # Normalize distance score
                        max_distance = 5000  # 5km maximum
                        spatial_score = float(np.exp(-dist / max_distance))
                    else:
                        spatial_score = 1.0
                        dist = 0
                    
                    # Combine scores
                    semantic_score = float(node.score)
                    combined_score = float(semantic_weight * semantic_score + spatial_weight * spatial_score)
                    
                    # Create a simplified node structure for visualization
                    reranked_node = {
                        'name': node.payload.get('name', ''),
                        'caption': node.payload.get('caption', ''),
                        'position': position,
                        'image_path': node.payload.get('image_path', ''),
                        'score': combined_score,
                        'semantic_score': semantic_score,
                        'spatial_score': spatial_score,
                        'distance': dist,
                        'type': node.payload.get('type', 'base'),
                        'original_id': {
                            'node_id': node.payload.get('original_id', {}).get('node_id'),
                            'graph_id': node.payload.get('original_id', {}).get('graph_id')
                        }
                    }
                    reranked_nodes.append(reranked_node)
                    
                except Exception as e:
                    print(f"Error processing node for reranking: {str(e)}")
                    continue
            
            # Sort by combined score
            reranked_nodes.sort(key=lambda x: x['score'], reverse=True)
            
            # Print reranked scores
            print("\nTop-k scores after spatial adjustment:")
            print(f"Weights: Semantic={semantic_weight:.2f}, Spatial={spatial_weight:.2f}")
            for i, node in enumerate(reranked_nodes[:10]):
                print(f"{i+1}. {node['name']}: {node['score']:.4f} "
                      f"(semantic: {node['semantic_score']:.4f}, "
                      f"spatial: {node['spatial_score']:.4f}, "
                      f"distance: {node['distance']:.1f}m)")
            
            # Generate context string from top-k reranked nodes
            context_nodes = reranked_nodes[:config.Config.RETRIEVAL['reranking']['top_k']]
            context_texts = ["Available locations with their hierarchical relationships:"]
            
            for base_node in context_nodes:
                node_id = base_node['original_id']['node_id']
                date_str = base_node['original_id']['graph_id']
                
                hierarchy_chain = self._get_node_hierarchy(node_id, date_str)
                
                context_texts.append("\nLocation Hierarchy:")
                
                # Start with highest level parent
                for i, node in enumerate(hierarchy_chain[:-1]):  # All except base node
                    indent = "  " * i
                    context_texts.append(f"{indent}Level {node['level']} Area:")
                    context_texts.append(f"{indent}{{")
                    context_texts.append(f"{indent}  \"name\": \"{node['name']}\",")
                    context_texts.append(f"{indent}  \"caption\": \"{node['caption']}\",")
                    context_texts.append(f"{indent}  \"type\": \"{node['type']}\",")
                    context_texts.append(f"{indent}  \"level\": {node['level']}")
                    context_texts.append(f"{indent}}} contains ↓")
                
                # Add base node with full details
                indent = "  " * (len(hierarchy_chain) - 1)
                context_texts.append(f"{indent}Base Location:")
                context_texts.append(f"{indent}{{")
                context_texts.append(f"{indent}  \"name\": \"{base_node['name']}\",")
                context_texts.append(f"{indent}  \"caption\": \"{base_node['caption']}\",")
                context_texts.append(f"{indent}  \"position\": {json.dumps(base_node['position'])},")
                context_texts.append(f"{indent}  \"image_path\": \"{base_node['image_path']}\",")
                context_texts.append(f"{indent}  \"score\": {base_node['score']:.4f},")
                context_texts.append(f"{indent}  \"parent_areas\": {json.dumps([n['name'] for n in hierarchy_chain[:-1]])}")
                context_texts.append(f"{indent}}}")
                context_texts.append("---")
            
            context = "\n".join(context_texts)
            print("============Context: ", context)  # Debug print
            
            if return_nodes:
                return context_nodes
            return context

        except Exception as e:
            print(f"Error in retrieve_hierarchical_context: {str(e)}")
            print(traceback.format_exc())
            return [] if return_nodes else ""

    def _get_node_hierarchy(self, base_node_id: str, date_str: str) -> List[Dict]:
        """Helper method to get the hierarchical chain of nodes"""
        node_chain = []
        current_node = base_node_id
        G = self.forests.get(date_str)
        
        if not G:
            return node_chain
        
        while current_node is not None:
            if current_node not in G.nodes:
                break
            
            node_data = G.nodes[current_node]
            node_chain.append({
                'name': node_data.get('name', ''),
                'caption': node_data.get('caption', ''),
                'type': node_data.get('type', 'base'),
                'level': node_data.get('level', 0),
                'node_id': current_node
            })
            
            # Find parent (node with higher level)
            parents = [n for n in G.neighbors(current_node) 
                      if G.nodes[n]['level'] > node_data['level']]
            current_node = parents[0] if parents else None
        
        return list(reversed(node_chain))

    def format_node_info(self, node_data: dict, node_id: str) -> str:
        """Format node information into readable text"""
        node_info = [
            f"\nLocation: {node_id}",
            f"Level: {node_data.get('level', 0)}",
            f"Type: {node_data.get('type', 'unknown')}"
        ]
        
        if node_data.get('type') == 'cluster':
            node_info.extend([
                f"Name: {node_data.get('name', '')}",
                f"Summary: {node_data.get('summary', '')}",
                f"Relationships: {node_data.get('relationships', '')}",
            ])
        else:
            # Base level node
            node_info.extend([
                f"Name: {node_data.get('name', '')}",
                f"caption: {node_data.get('caption', '')}",
                f"image_path: {node_data.get('image_path', '')}"
            ])
        
        return "\n".join(filter(None, node_info))
    
    async def generate_response(self, query: str, context: str) -> str:
        """Generate a response based on the query and retrieved context"""
        
        history_length = 3
        recent_history = ""
        if len(self.chat_history) >= history_length:
            for chat in self.chat_history[-2:]:
                recent_history += f"\nUser: {chat['user_input']}\nAssistant: {chat['response']}\n"
        
        prompt = f"""As an environmental analysis expert, answer the following query using the provided context information:

        Recent Conversation History:
        {recent_history}

        Current Query: {query}

        Context Information:
        {context}

        Instructions:
        1. If the query is asking for a specific location, provide the best one answer based on the context, in the EXACT following parseable format: 
            1.1 Choose the SINGLE best location from the base locations that best matches the query
            1.2 Consider the entire hierarchical context - a location's parent areas may provide important context
            1.3 Return your answer in the EXACT following JSON format:
            {{
                "name": "Best matching base location name",
                "caption": "Best matching base location caption",
                "position": {{
                    "x": "Best matching x coordinate",
                    "y": "Best matching y coordinate"
                }},
                "image_path": "Best matching image_path",
                "parent_areas": ["List of parent area names"],
                "reasons": "Explain why this location is the best match, including how its parent areas contribute to the decision"
            }}
        2. If the query is asking for a general environmental analysis, provide a detailed analysis of the environment based on the context.
        """
        
        return await self.llm.generate_response(prompt)

    async def generate_response_no_history(self, query: str, context: str) -> str:
        """Generate a response without considering chat history"""
        # Use double curly braces to escape JSON template
        prompt = f"""As an environmental analysis expert, answer the following query using the provided context information:
        
        Current Query: {query}

        Context Information:
        {context}

        Instructions:
        1. If the query is asking for a specific location, provide the best one answer based on the context, in the EXACT following parseable format: 
            1.1 Choose the SINGLE best location from the base locations that best matches the query
            1.2 Consider the entire hierarchical context - a location's parent areas may provide important context
            1.3 Return your answer in the EXACT following JSON format:
            {{
                "name": "Best matching base location name",
                "caption": "Best matching base location caption",
                "position": {{
                    "x": "Best matching x coordinate",
                    "y": "Best matching y coordinate"
                }},
                "image_path": "Best matching image_path",
                "parent_areas": ["List of parent area names"],
                "reasons": "Explain why this location is the best match, including how its parent areas contribute to the decision"
            }}


        2. If the query is asking for a general environmental analysis, provide a detailed analysis of the environment based on the context.
        """
        
        print("============Context: ", context)
        return await self.llm.generate_response(prompt)

    async def chat(self):
        """Interactive chat interface with support for different input scenarios"""
        print("\nWelcome to the Embodied RAG!")
        print("\nCommands:")
        print("- 'quit': Exit the chat")
        print("- 'clear': Clear chat history")
        print("\nInput formats:")
        print("1. Query only: Just type your question")
        print("2. Query + Location: Use format 'L: lat,lon | Q: your question'")
        print("3. Query + Location + History: Use format 'L: lat,lon | H: true | Q: your question'")
        
        if not self.forests:
            print("\nError: No forests were loaded.")
            return
        
        while True:
            try:
                user_input = input("\nYou: ").strip()
                
                if user_input.lower() == 'quit':
                    break
                elif user_input.lower() == 'clear':
                    self.chat_history = []
                    print("Chat history cleared.")
                    continue
                
                # Parse input format
                query = user_input
                agent_location = None
                use_history = False
                
                if '|' in user_input:
                    parts = [p.strip() for p in user_input.split('|')]
                    for part in parts:
                        if part.startswith('L:'):
                            try:
                                lat, lon = map(float, part[2:].strip().split(','))
                                agent_location = {'latitude': lat, 'longitude': lon}
                            except:
                                print("Invalid location format. Expected 'L: latitude,longitude'")
                                continue
                        elif part.startswith('H:'):
                            use_history = part[2:].strip().lower() == 'true'
                        elif part.startswith('Q:'):
                            query = part[2:].strip()
                
                # Get context based on input scenario
                context = await self.retrieve_hierarchical_context(
                    query=query,
                    agent_location=agent_location
                )
                
                # Generate response with or without history
                if use_history:
                    response = await self.generate_response(query, context)
                else:
                    response = await self.generate_response_no_history(query, context)
                
                print("\nAssistant:", response)
                
                # Record the interaction if using history
                if use_history:
                    self.chat_history.append({
                        'user_input': query,
                        'context': context,
                        'response': response,
                        'location': agent_location
                    })
                    
            except Exception as e:
                print(f"\nError: {str(e)}")
                print(traceback.format_exc())
                continue

    async def index_graph_nodes(self, G: nx.Graph, date_str: str):
        """Index all nodes in the graph to the vector database"""
        print("\nIndexing nodes to vector database...")
        total_nodes = len(G.nodes())
        
        # Clear existing points
        print("Clearing existing points...")
        self.vector_db.delete(
            collection_name=self.collection_name,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="date",
                            match=models.MatchValue(value=date_str)
                        )
                    ]
                )
            )
        )
        
        # Prepare nodes for batch indexing
        nodes_to_index = []
        failed_nodes = []
        
        print(f"\nIndexing {total_nodes} nodes...")
        progress_bar = tqdm(G.nodes(data=True), total=total_nodes)
        
        for node_id, node_data in progress_bar:
            # Format node text for embedding
            node_text = self.format_node_info(node_data, node_id)
            
            # Generate embedding
            try:
                embedding_response = await self.llm.client.embeddings.create(
                    model="text-embedding-ada-002",
                    input=node_text
                )
                embedding = embedding_response.data[0].embedding
                
                # Create point
                point = models.PointStruct(
                    id=str(uuid.uuid4()),
                    vector=embedding,
                    payload={
                        "original_id": {
                            "node_id": node_id,
                            "graph_id": date_str
                        },
                        "date": date_str,
                        # Flatten node_data into root level
                        "name": node_data.get('name', ''),
                        "caption": node_data.get('caption', ''),
                        "position": node_data.get('position', {}),
                        "image_path": node_data.get('image_path', ''),
                        "type": node_data.get('type', 'base'),
                        "level": node_data.get('level', 0),
                        "text": node_text
                    }
                )
                nodes_to_index.append(point)
                
            except Exception as e:
                failed_nodes.append((node_id, str(e)))
                progress_bar.write(f"Error indexing node {node_id}: {str(e)}")
                continue
            
            # Batch upload every 100 nodes
            if len(nodes_to_index) >= 100:
                try:
                    self.vector_db.upsert(
                        collection_name=self.collection_name,
                        points=nodes_to_index
                    )
                    progress_bar.write(f"✓ Indexed batch of {len(nodes_to_index)} nodes")
                except Exception as e:
                    progress_bar.write(f"Error uploading batch: {str(e)}")
                    failed_nodes.extend([(n.payload['original_id']['node_id'], "Batch upload failed") for n in nodes_to_index])
                nodes_to_index = []
                
                # Add small delay to avoid rate limits
                await asyncio.sleep(0.1)
        
        # Index remaining nodes
        if nodes_to_index:
            try:
                self.vector_db.upsert(
                    collection_name=self.collection_name,
                    points=nodes_to_index
                )
                progress_bar.write(f"✓ Indexed final {len(nodes_to_index)} nodes")
            except Exception as e:
                progress_bar.write(f"Error uploading final batch: {str(e)}")
                failed_nodes.extend([(n.payload['original_id']['node_id'], "Final batch upload failed") for n in nodes_to_index])
        
        total_points = self.vector_db.count(self.collection_name)
        print(f"\nIndexing complete:")
        print(f"- Total nodes in vector database: {total_points}")
        print(f"- Successfully indexed: {total_points} nodes")
        print(f"- Failed to index: {len(failed_nodes)} nodes")
        
        if failed_nodes:
            print("\nFailed nodes:")
            for node_id, error in failed_nodes[:10]:  # Show first 10 failures
                print(f"- Node {node_id}: {error}")
            if len(failed_nodes) > 10:
                print(f"... and {len(failed_nodes) - 10} more")

def main():
    print("Starting Environmental Analysis Assistant...")
    try:
        async def init_and_chat():
            chatbot = await EnvironmentalChat.create()
            await chatbot.chat()
        
        asyncio.run(init_and_chat())
        print("Chat session ended")
    except Exception as e:
        print(f"Fatal error: {str(e)}")
        print(f"Stack trace: {traceback.format_exc()}")

if __name__ == "__main__":
    main() 