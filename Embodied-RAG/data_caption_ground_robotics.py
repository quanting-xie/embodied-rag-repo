import json
import networkx as nx
from datetime import datetime
import base64
from typing import Dict, Any
from pathlib import Path
import sys
import os
from openai import AsyncOpenAI
import traceback

# Add Embodied_RAG to Python path
current_dir = Path(__file__).parent.absolute()
embodied_rag_dir = current_dir.parent / "Embodied_RAG"
sys.path.append(str(embodied_rag_dir))

from config import Config
import asyncio

class JsonInterpreter:
    def __init__(self, dataset_base_dir: str = "/Users/danielxie/Embodied-RAG_datasets"):
        self.graph = nx.Graph()
        self.node_counter = 0
        self.data_dir = Path(__file__).parent.absolute()
        self.graph_dir = self.data_dir / "graph"
        self.dataset_base_dir = Path(dataset_base_dir)
        
        # Get OpenAI API key from environment variable
        self.api_key = os.getenv('OPENAI_API_KEY')
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY environment variable is not set")
        
        self.client = AsyncOpenAI(api_key=self.api_key)
        self.model = Config.LLM['model']
        self.temperature = Config.LLM['temperature']
        self.max_tokens = Config.LLM['max_tokens']
        
        self.graph_dir.mkdir(parents=True, exist_ok=True)

    def get_image_path(self, node_data: Dict, dataset: str) -> Path:
        """Construct the correct image path for a given node"""
        # Use the image_path directly from the node data
        return self.dataset_base_dir / node_data['image_path']

    def read_json_file(self, json_path: str) -> Dict[str, Any]:
        """Read and parse the JSON file containing robot capture data"""
        json_path = Path(json_path)
        print(f"Reading from: {json_path}")
        
        if not json_path.exists():
            raise FileNotFoundError(f"JSON file not found: {json_path}")
            
        with open(json_path, 'r') as file:
            try:
                data = json.load(file)
                # Extract nodes from the data structure
                nodes = data.get('data', [])  # Changed from 'nodes' to 'data'
                print(f"Found {len(nodes)} capture points")
                return nodes
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON: {e}")
                raise

    async def prepare_batch_requests(self, nodes: list, dataset: str, batch_size: int = 100) -> list:
        """Prepare batch requests for image captioning in smaller batches"""
        all_batch_requests = []
        total_nodes = len(nodes)
        
        for i in range(0, total_nodes, batch_size):
            batch_nodes = nodes[i:i + batch_size]
            batch_requests = []
            
            print(f"Preparing batch {i//batch_size + 1}/{(total_nodes + batch_size - 1)//batch_size} "
                  f"({i}/{total_nodes} nodes processed)")
            
            for node_index, node in enumerate(batch_nodes):
                # Generate a node ID based on the batch index and node index
                node_id = str(i + node_index)  # This ensures unique IDs across batches
                
                # Use the correct image path
                image_path = self.get_image_path(node, dataset)
                
                if not image_path.exists():
                    print(f"Skipping node {node_id}: Image not found at {image_path}")
                    continue
                    
                try:
                    # Read and encode image
                    with open(image_path, "rb") as image_file:
                        image_data = base64.b64encode(image_file.read()).decode('utf-8')
                    
                    # Create batch request
                    request = {
                        "custom_id": node_id,  # Use our generated ID
                        "method": "POST",
                        "url": "/v1/chat/completions",
                        "body": {
                            "model": self.model,
                            "messages": [
                                {
                                    "role": "system",
                                    "content": "You are an grass analyst. Analyze this image and describe the environment."
                                },
                                {
                                    "role": "user",
                                    "content": [
                                        {
                                            "type": "text",
                                            "text": """Analyze this image:
                                            1. A short abstract name (4-6 words) summarize the grass condition
                                            2. A detailed description of the environment, including:
                                               - List objects/elements in the scene
                                               - List the grass condition
                                               - List weeds and other plants

                                            Format your response as:
                                            NAME: <4-6 word abstract name>, use only nouns.
                                            DESCRIPTION: <detailed grass condition description>"""
                                        },
                                        {
                                            "type": "image_url",
                                            "image_url": {
                                                "url": f"data:image/jpeg;base64,{image_data}"
                                            }
                                        }
                                    ]
                                }
                            ],
                            "temperature": self.temperature,
                            "max_tokens": self.max_tokens
                        }
                    }
                    batch_requests.append(request)
                except Exception as e:
                    print(f"Error processing node {node_id}: {str(e)}")
                    continue
            
            all_batch_requests.extend(batch_requests)
            print(f"Added {len(batch_requests)} requests to batch")
            
            await asyncio.sleep(0.1)
        
        return all_batch_requests

    async def process_batch_results(self, results: list) -> dict:
        """Process batch results into caption data"""
        captions = {}
        for result in results:
            custom_id = result['custom_id']
            if result.get('error'):
                captions[custom_id] = {
                    "name": "Error Location",
                    "caption": f"Error generating caption: {result['error']}"
                }
                continue
                
            content = result['response']['body']['choices'][0]['message']['content'].strip()
            name = ""
            description = ""
            
            # More robust parsing that handles markdown and variations
            for line in content.split('\n'):
                line = line.strip()
                # Handle various name formats
                if any(prefix in line.upper() for prefix in ['NAME:', '**NAME:**', '*NAME:*']):
                    name = line.replace('NAME:', '').replace('*', '').strip()
                # Handle various description formats
                elif any(prefix in line.upper() for prefix in ['DESCRIPTION:', '**DESCRIPTION:**', '*DESCRIPTION:*']):
                    description = line.replace('DESCRIPTION:', '').replace('*', '').strip()
            
            captions[custom_id] = {
                "name": name if name else "Unnamed Street",
                "caption": description if description else content
            }
        
        return captions

    async def process_json_data(self, json_path: str, output_path: str, dataset: str):
        try:
            nodes = self.read_json_file(json_path)
            
            # Process in parallel chunks
            chunk_size = 100
            chunks = [nodes[i:i + chunk_size] for i in range(0, len(nodes), chunk_size)]
            
            # Create tasks for parallel processing
            tasks = []
            for chunk_index, chunk in enumerate(chunks):
                task = self.process_chunk(chunk, chunk_index, dataset)
                tasks.append(task)
            
            # Process all chunks in parallel
            print(f"Processing {len(chunks)} chunks in parallel...")
            results = await asyncio.gather(*tasks)
            
            # Combine results
            for chunk_result in results:
                if chunk_result:
                    for node_id, node_data in chunk_result.items():
                        self.graph.add_node(
                            node_id,
                            **node_data
                        )
            
            # Save final graph
            output_path = Path(output_path)
            if not output_path.is_absolute():
                output_path = self.graph_dir / output_path
            
            nx.write_gml(self.graph, output_path)
            print(f"Graph saved to {output_path}")
            
        except Exception as e:
            print(f"Error in process_json_data: {str(e)}")
            print(traceback.format_exc())

    async def process_chunk(self, chunk_nodes: list, chunk_index: int, dataset: str) -> dict:
        """Process a single chunk of nodes"""
        try:
            print(f"\nProcessing chunk {chunk_index} (size: {len(chunk_nodes)})")
            
            # Prepare batch requests
            batch_requests = await self.prepare_batch_requests(chunk_nodes, dataset)
            if not batch_requests:
                return {}

            # Save batch requests to JSONL
            batch_input_path = self.data_dir / f"batch_input_{chunk_index}.jsonl"
            with open(batch_input_path, "w") as f:
                for request in batch_requests:
                    f.write(json.dumps(request) + "\n")

            # Upload and create batch
            batch_file = await self.client.files.create(
                file=open(batch_input_path, "rb"),
                purpose="batch"
            )
            
            batch = await self.client.batches.create(
                input_file_id=batch_file.id,
                endpoint="/v1/chat/completions",
                completion_window="24h"
            )
            
            # Monitor batch status
            while True:
                batch_status = await self.client.batches.retrieve(batch.id)
                status = batch_status.status
                print(f"Chunk {chunk_index} status: {status}")
                
                if status in ['completed', 'failed', 'expired']:
                    break
                await asyncio.sleep(30)
            
            if batch_status.status == 'completed':
                # Process results
                output_file = await self.client.files.content(batch_status.output_file_id)
                results = [json.loads(line) for line in output_file.text.split('\n') if line]
                captions = await self.process_batch_results(results)
                
                # Create node data with correct position extraction
                node_data = {}
                for i, node in enumerate(chunk_nodes):
                    node_id = str(i)  # Generate sequential IDs
                    caption_data = captions.get(node_id, {"name": "No Location", "caption": "No caption available"})
                    
                    # Extract coordinates from the gps_location object
                    try:
                        gps = node.get('gps_location', {})
                        # Note: x is latitude, y is longitude in your data
                        latitude = float(gps['x'])
                        longitude = float(gps['y'])
                        
                        node_data[node_id] = {
                            "type": "robot_capture",
                            "image_path": node['image_path'],
                            "name": caption_data["name"],
                            "caption": caption_data["caption"],
                            "position": {
                                'x': longitude,  # Convert to match CMU format
                                'y': latitude,   # Convert to match CMU format
                                'z': float(gps.get('z', 0.0))
                            },
                            "timestamp": node.get('timestamp', ''),
                            "level": 0,
                            "ndvi": node.get('NDVI', 0.0),
                            "temperature": node.get('temperature', None)
                        }
                    except (KeyError, TypeError, ValueError) as e:
                        print(f"Error extracting coordinates for node {node_id}: {str(e)}")
                        continue
                
                # Clean up
                batch_input_path.unlink()
                return node_data
                
        except Exception as e:
            print(f"Error processing chunk {chunk_index}: {str(e)}")
            print(traceback.format_exc())
            return {}

async def main():
    interpreter = JsonInterpreter()
    
    try:
        # Process the robot capture JSON file
        dataset = "capture_session_20250113_161614"
        json_path = f"/Users/danielxie/Embodied-RAG_datasets/{dataset}/metadata/capture_data.json"
        output_path = f"{dataset}_graph.gml"
        
        # Verify image directory exists
        image_dir = interpreter.dataset_base_dir / dataset
        if not image_dir.exists():
            raise FileNotFoundError(f"Image directory not found: {image_dir}")
            
        print(f"Using image directory: {image_dir}")
        
        await interpreter.process_json_data(json_path, output_path, dataset)
        print("\nProcessing complete!")
            
    except Exception as e:
        print(f"Error during processing: {str(e)}")
        print(traceback.format_exc())

if __name__ == "__main__":
    asyncio.run(main())