from typing import List, Dict, Tuple
import numpy as np
from pathlib import Path
import asyncio
from retrieval_and_generation import EnvironmentalChat
import json
from tqdm import tqdm
from datetime import datetime
import googlemaps
import os
import scipy.stats
import folium
from folium import plugins
import webbrowser
import uuid
import base64
import argparse
import traceback
from utils.spatial import compute_spatial_score, normalize_location_format, compute_haversine_distance
import time
import networkx as nx
from llm import LLMInterface
from llm_retrieval import ParallelLLMRetriever
from config import Config
        
class RetrievalEvaluator:
    def __init__(self, graph_path: str = None, vector_db_path: str = None, image_dir: str = None):
        self.chat = None
        self.query_locations_file = Path("query_locations.json")
        self.results_dir = Path(__file__).parent.absolute() / "evaluation_results"
        self.results_dir.mkdir(exist_ok=True)
        self.k = Config.RETRIEVAL['search_params']['k-branch']  # Top-k results to compare
        
        # Set paths from arguments or use defaults, ensuring absolute paths
        self.graph_path = Path(graph_path).resolve() 
        self.vector_db_path = Path(vector_db_path).resolve()
        self.image_dir = Path(image_dir).resolve() 
        
        print(f"Using graph: {self.graph_path}")
        print(f"Using vector database: {self.vector_db_path}")
        print(f"Using images from: {self.image_dir}")
        
        # Validate paths
        if not self.graph_path.exists():
            raise FileNotFoundError(f"Graph file not found: {self.graph_path}")
        
        # Create directories if they don't exist
        self.vector_db_path.mkdir(parents=True, exist_ok=True)
        
        # Use default center from config instead of hardcoding
        self.default_center = Config.LOCATION['default_center']
        self.radius = Config.LOCATION['search_radius']  # Also move radius to config
        
        # Enhanced vector_db path validation
        if self.vector_db_path.exists():
            files = list(self.vector_db_path.glob('**/*'))  # Recursively list all files
            print("\nVector Database Directory Contents:")
            print(f"Path: {self.vector_db_path}")
            print(f"Total files found: {len(files)}")
            for f in files:
                print(f"  - {f.relative_to(self.vector_db_path)}")
                if f.is_file():
                    print(f"    Size: {f.stat().st_size:,} bytes")
        
        # Initialize both retrievers
        self.chat = None  # Original retriever
        self.parallel_retriever = None  # New parallel retriever
        
        self.experiment_log_dir = Path(__file__).parent.absolute() / "experiment_logs"
        self.experiment_log_dir.mkdir(exist_ok=True)
        self.current_experiment_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        
    async def initialize(self):
        """Initialize both retrieval systems"""
        # Update config for paths
        import config
        config.Config.PATHS = {
            'graph_path': str(self.graph_path.resolve()),
            'vector_db_path': str(self.vector_db_path.resolve()),
            'image_dir': str(self.image_dir.resolve())
        }
        
        # Initialize original retriever
        self.chat = await EnvironmentalChat.create()
        

        # Load graph
        graph = nx.read_gml(self.graph_path)
        llm_interface = LLMInterface()
        self.parallel_retriever = ParallelLLMRetriever(graph, llm_interface)

    async def interactive_mode(self):
        """Modified interactive mode to support method selection"""
        print("\n=== Interactive Query Mode ===")
        print("\nRetrieval Methods:")
        print("1. Original")
        print("2. Parallel LLM")
        
        method = input("\nSelect retrieval method (1/2): ").strip()
        retrieval_method = 'original' if method == '1' else 'parallel'
        
        print("\nInput styles:")
        print("1. Query only (e.g., 'Where are the convenience stores?')")
        print("2. Query + Location (use format 'L: lat,lon | Q: your question')")
        print("3. Query + Location + History (use format 'L: lat,lon | H: true | Q: your question')")
        print("\nCommands:")
        print("- 'quit': Exit the chat")
        print("- 'clear': Clear chat history")
        
        # Get user input
        user_input = input("\nEnter your query: ").strip()
        
        if user_input.lower() == 'quit':
            return False
        elif user_input.lower() == 'clear':
            self.chat_history = []
            print("Chat history cleared.")
            return True
        
        # Parse input format
        query = user_input
        location = self.default_center
        use_history = False
        
        if '|' in user_input:
            parts = [p.strip() for p in user_input.split('|')]
            for part in parts:
                if part.startswith('L:'):
                    try:
                        lat, lon = map(float, part[2:].strip().split(','))
                        location = {
                            'latitude': lat,
                            'longitude': lon
                        }
                    except:
                        print("Invalid location format. Using default location (Tokyo).")
                elif part.startswith('H:'):
                    use_history = part[2:].strip().lower() == 'true'
                elif part.startswith('Q:'):
                    query = part[2:].strip()
        
        # Create query item
        query_item = {
            'query': query,
            'location': location,
            'use_history': use_history,
            'searchable_term': query.split()[-1]  # Simple extraction of search term
        }
        
        # Update query evaluation
        result = await self.evaluate_query(query_item, retrieval_method=retrieval_method)
        
        return True

    def visualize_results(self, query_item: Dict, retrieved_nodes: List[Dict], ground_truth: List[Dict]):
        """Visualize query results on an interactive map with clickable image popups"""
        # Create map centered on query location
        center_lat = query_item['location']['latitude']
        center_lon = query_item['location']['longitude']
        m = folium.Map(location=[center_lat, center_lon], zoom_start=16)
        
        # Add marker for query location
        folium.Marker(
            location=[center_lat, center_lon],
            popup=f"Query Location\n{query_item['query']}",
            icon=folium.Icon(color='red', icon='info-sign')
        ).add_to(m)
        
        # Add retrieved results with image popups
        for i, node in enumerate(retrieved_nodes):
            try:
                # Skip nodes without position data
                if not node.get('position'):
                    print(f"Skipping node {i} - no position data")
                    continue
                    
                # Get position data - handle both x/y and lat/lon formats
                position = node['position']
                if not isinstance(position, dict):
                    print(f"Skipping node {i} - invalid position format")
                    continue
                    
                # Try to get coordinates, skipping if not found
                try:
                    lat = float(position.get('y', position.get('latitude')))
                    lon = float(position.get('x', position.get('longitude')))
                except (TypeError, ValueError):
                    print(f"Skipping node {i} - invalid coordinate values")
                    continue
                    
                # Get node metadata
                name = node.get('name', 'Unnamed')
                caption = node.get('caption', 'No caption available')
                
                # Handle image path
                image_path = node.get('image_path', '')
                encoded_image = None
                
                if image_path:
                    # Use consistent path resolution
                    image_name = Path(image_path).name
                    absolute_image_path = self.image_dir / image_name  # Use self.image_dir directly
                    
                    print(f"Trying to load image from: {absolute_image_path}")
                    
                    # Try to read the image if path exists
                    if absolute_image_path.exists():
                        try:
                            with open(absolute_image_path, 'rb') as img_file:
                                encoded_image = base64.b64encode(img_file.read()).decode()
                        except Exception as e:
                            print(f"Error reading image file {absolute_image_path}: {str(e)}")
                    else:
                        print(f"Image not found at: {absolute_image_path}")
                
                # Create popup HTML
                popup_html = f"""
                <div style="width:800px; max-height:800px; overflow-y:auto;">
                    <h3 style="margin-bottom:10px;">Retrieved Result #{i+1}</h3>
                    <div style="margin-bottom:15px;">
                        <b>Name:</b> {name}<br>
                        <b>Score:</b> {node.get('score', 'N/A')}
                    </div>
                """
                
                if encoded_image:
                    popup_html += f"""
                    <div style="margin-bottom:15px;">
                        <img src="data:image/jpeg;base64,{encoded_image}" 
                             style="width:100%; max-width:800px; height:auto; margin-bottom:10px;">
                    </div>
                    """
                
                popup_html += f"""
                    <div style="margin-bottom:15px;">
                        <b>Caption:</b><br>
                        <div style="white-space:pre-wrap; padding:10px; background:#f5f5f5; border-radius:5px;">
                            {caption}
                        </div>
                    </div>
                    <div>
                        <b>Location:</b> {lat:.6f}, {lon:.6f}
                    </div>
                </div>
                """
                
                # Create marker with popup
                folium.Marker(
                    location=[lat, lon],
                    popup=folium.Popup(folium.Html(popup_html, script=True), max_width=800),
                    icon=folium.Icon(color='blue', icon='info-sign')
                ).add_to(m)
                
            except Exception as e:
                print(f"Error processing node {i}:")
                print(f"Error: {str(e)}")
                print(f"Node data: {node}")
                continue
        
        # Add ground truth locations
        for i, place in enumerate(ground_truth):
            try:
                lat = place['position']['latitude']
                lon = place['position']['longitude']
                folium.Marker(
                    location=[lat, lon],
                    popup=f"Ground Truth #{i+1}\n{place['name']}",
                    icon=folium.Icon(color='green', icon='info-sign')
                ).add_to(m)
            except Exception as e:
                print(f"Error adding ground truth marker {i}: {str(e)}")
        
        # Add legend
        legend_html = """
        <div style="position: fixed; bottom: 50px; left: 50px; z-index: 1000; background-color: white; 
             padding: 10px; border: 2px solid grey; border-radius: 5px">
        <p><i class="fa fa-map-marker fa-2x" style="color:red"></i> Query Location</p>
        <p><i class="fa fa-map-marker fa-2x" style="color:blue"></i> Retrieved Results</p>
        <p><i class="fa fa-map-marker fa-2x" style="color:green"></i> Ground Truth</p>
        </div>
        """
        m.get_root().html.add_child(folium.Element(legend_html))
        
        # Save and open map
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        map_file = self.results_dir / f"results_map_{timestamp}.html"
        m.save(str(map_file))
        webbrowser.open(f'file://{map_file}')

    def compute_spatial_relativity(self, query_location: Dict, retrieved_nodes: List[Dict]) -> Dict:
        """Compute spatial relativity and parent coverage metrics"""
        if not retrieved_nodes:
            return {
                'score': 0.0,
                'top1_spatial': 0.0,
                'top5_spatial': 0.0,
                'level1_coverage': 0.0
            }

        # Get the graph from chat instance
        graph = self.chat.forests[list(self.chat.forests.keys())[0]]
        distances = []
        level1_parents = set()
        
        # Collect distances and level 1 parents for top 5 nodes
        for node in retrieved_nodes[:self.k]:
            try:
                if 'position' in node:
                    node_loc = normalize_location_format(node['position'])
                    # Calculate raw distance in meters
                    distance = compute_haversine_distance(query_location, node_loc)
                    # Normalize distance to 0-1 score (closer = higher score)
                    spatial_score = max(0, 1 - (distance / self.radius))
                    distances.append(spatial_score)
                    
                    # Find level 1 parents (keeping this part for comprehensiveness)
                    base_node_id = None
                    for n in graph.nodes():
                        node_data = graph.nodes[n]
                        if (node_data.get('level') == 0 and 
                            node_data.get('name') == node.get('name')):
                            base_node_id = n
                            break

                    if base_node_id:
                        for n in graph.nodes():
                            node_data = graph.nodes[n]
                            if node_data.get('level') == 1:
                                members = node_data.get('members', [])
                                if str(base_node_id) in members:
                                    level1_parents.add(n)
                
            except Exception as e:
                print(f"Error processing node: {str(e)}")
                traceback.print_exc()
                continue

        # Calculate coverage ratio
        total_level1_nodes = sum(1 for n in graph.nodes() if graph.nodes[n].get('level') == 1)
        unique_parents = len(level1_parents)
        coverage_ratio = unique_parents / total_level1_nodes if total_level1_nodes > 0 else 0.0

        return {
            'top1_spatial': distances[0] if distances else 0.0,  # Spatial score for top 1
            'top5_spatial': sum(distances) / len(distances) if distances else 0.0,  # Average spatial score for top 5
            'level1_coverage': coverage_ratio,
            'unique_level1_parents': unique_parents,
            'total_level1_nodes': total_level1_nodes,
            'raw_spatial_scores': distances  # Keep raw scores for potential use
        }

    async def compute_semantic_relativity(self, query: str, retrieved_nodes: List[Dict]) -> Dict[str, float]:
        """Compute semantic relativity and score variability"""
        system_prompt = """You are an expert evaluator. Rate the relevance of the location given a user's query on a scale of 0-100, where:

        Consider:
        1. How well the location matches the query intent
        2. The relevance of the visual content
        3. The location's hierarchical context
        4. The accuracy and completeness of the match
        
        Return only the numerical score without explanation."""
        
        async def get_score_for_node(node: Dict, node_index: int) -> float:
            """Get average score for a single node with multiple attempts"""
            node_scores = []
            num_attempts = 5
            
            for attempt in range(num_attempts):
                try:
                    # Get image path
                    image_path = node.get('image_path', '')
                    if not image_path:
                        print(f"Node {node_index}, Attempt {attempt + 1}: No image path available")
                        continue

                    # Use consistent path resolution
                    image_name = Path(image_path).name
                    absolute_image_path = self.image_dir / image_name
                    
                    if not absolute_image_path.exists():
                        print(f"Node {node_index}, Attempt {attempt + 1}: Image not found at {absolute_image_path}")
                        continue

                    # Read and encode image
                    try:
                        with open(absolute_image_path, 'rb') as img_file:
                            encoded_image = base64.b64encode(img_file.read()).decode()
                    except Exception as e:
                        print(f"Node {node_index}, Attempt {attempt + 1}: Error reading image file: {str(e)}")
                        continue

                    # Format location with image and hierarchical context
                    location_text = (
                        f"Location Name: {node.get('name', 'Unnamed')}\n"
                        f"Parent Areas: {node.get('parent_areas', [])}\n"
                        f"Visual Content: [Image Attached]\n"
                        f"Description: {node.get('caption', 'No description')}"
                    )

                    prompt = f"""Query: {query}

Location Information:
{location_text}

Rate the relevance of this location on a scale of 0-100:"""

                    # Get score from GPT-4 with image
                    response = await self.chat.llm.generate_response(
                        prompt, 
                        system_prompt,
                        image_base64=encoded_image
                    )
                    
                    try:
                        score = float(response.strip())
                        if 0 <= score <= 100:
                            print(f"Node {node_index}, Attempt {attempt + 1}: Score = {score}")
                            node_scores.append(score)
                        else:
                            print(f"Node {node_index}, Attempt {attempt + 1}: Invalid score range: {score}")
                    except ValueError:
                        print(f"Node {node_index}, Attempt {attempt + 1}: Invalid response format: {response}")
                    
                except Exception as e:
                    print(f"Node {node_index}, Attempt {attempt + 1}: Error: {str(e)}")
                    traceback.print_exc()
            
            # Store raw scores in the node for later std calculation
            node['_raw_scores'] = node_scores
            return sum(node_scores) / len(node_scores) if node_scores else 0.0

        # Evaluate all nodes in parallel
        print("\nEvaluating nodes in parallel...")
        tasks = [
            get_score_for_node(node, i+1) 
            for i, node in enumerate(retrieved_nodes[:5])  # Only evaluate top 5
        ]
        
        scores_per_node = await asyncio.gather(*tasks)
        
        # Calculate standard deviation for top1 node (across its 5 attempts)
        all_raw_scores = []  # Store all raw scores for top5 std calculation
        top1_scores = []
        
        for i, score in enumerate(scores_per_node, 1):
            node_raw_scores = retrieved_nodes[i-1].get('_raw_scores', [])  # Get raw scores from node
            all_raw_scores.extend(node_raw_scores)  # Add to all scores for top5 std
            if i == 1:  # Top 1 node
                top1_scores = node_raw_scores
        
        # Calculate standard deviations
        top1_std = np.std(top1_scores) if top1_scores else 0.0
        top5_std = np.std(all_raw_scores) if all_raw_scores else 0.0
        
        print(f"\nScore Variability Metrics:")
        print(f"Top 1 Score Std: {top1_std:.2f}")
        print(f"Top 5 Score Std: {top5_std:.2f}")
        
        if not scores_per_node:
            return {'top1': 0.0, 'top5': 0.0, 'top1_std': 0.0, 'top5_std': 0.0}
        
        # Normalize scores to 0-1 range
        normalized_scores = [score / 100.0 for score in scores_per_node]
        
        return {
            'top1': normalized_scores[0] if normalized_scores else 0.0,
            'top5': sum(normalized_scores) / len(normalized_scores) if normalized_scores else 0.0,
            'raw_scores': scores_per_node,
            'top1_std': top1_std / 100.0,  # Normalize to 0-1 range
            'top5_std': top5_std / 100.0   # Normalize to 0-1 range
        }

    async def evaluate_generated_response(self, query: str, generated_response: str, retrieved_nodes: List[Dict]) -> Dict:
        """Evaluate the generated response using LLM to determine response type"""
        try:
            # Use LLM to determine response type
            system_prompt = """You are an expert at analyzing text responses. 
            Determine if the given response is location-specific (contains specific place details, coordinates, or location descriptions) 
            or general text (generic information or non-location-specific answer).
            
            Return only 'location' or 'general' without explanation."""
            
            classify_prompt = f"""Query: {query}

Response to analyze:
{generated_response}

Is this response location-specific or general text?"""

            response_type = await self.chat.llm.generate_response(classify_prompt, system_prompt)
            is_location_specific = response_type.strip().lower() == 'location'

            print("\n=== Evaluating Generated Response ===")
            print(f"Response Type: {'Location-specific' if is_location_specific else 'General text'}")
            
            if is_location_specific:
                # Try to extract location details using LLM
                extract_prompt = """Extract the following location details from the response as JSON:
                {
                    "name": "location name",
                    "caption": "location description",
                    "position": {"latitude": 0.0, "longitude": 0.0},
                    "image_path": "path if mentioned",
                    "parent_areas": ["parent area names"],
                    "reasons": "reasons for recommendation"
                }
                Return only valid JSON."""
                
                location_json = await self.chat.llm.generate_response(generated_response, extract_prompt)
                try:
                    # Clean up the response by removing markdown code block markers
                    cleaned_json = location_json.strip()
                    if cleaned_json.startswith('```'):
                        # Remove opening markdown
                        cleaned_json = cleaned_json.split('\n', 1)[1]
                    if cleaned_json.endswith('```'):
                        # Remove closing markdown
                        cleaned_json = cleaned_json.rsplit('\n', 1)[0]
                    # Remove any "json" language identifier
                    cleaned_json = cleaned_json.replace('json\n', '', 1)
                    
                    generated_node = json.loads(cleaned_json)
                except json.JSONDecodeError:
                    print("Failed to parse location details, falling back to general evaluation")
                    print(f"Attempted to parse JSON: {cleaned_json}")
                    is_location_specific = False
                
                if is_location_specific:
                    # Rest of location-specific evaluation
                    print(f"Generated Location: {generated_node.get('name', 'Unknown')}")
                    
                    # Get semantic score (single call, internally does 5 attempts)
                    score = await self.compute_semantic_relativity(query, [generated_node])
                    avg_semantic_score = score['top1'] if score['top1'] is not None else 0.0
                    
                    # Get spatial score if coordinates are available
                    spatial_score = 0.0
                    if generated_node.get('position'):
                        node_loc = normalize_location_format(generated_node['position'])
                        spatial_score = compute_spatial_score(
                            {'latitude': float(node_loc['latitude']), 
                             'longitude': float(node_loc['longitude'])}, 
                            retrieved_nodes
                        )
                    

                    # Calculate combined score
                    final_score = avg_semantic_score * spatial_score
                    
                    generation_evaluation = {
                        'semantic_score': avg_semantic_score,
                        'spatial_score': spatial_score,
                        'combined_score': final_score,
                        'is_location_specific': True,
                        'location_details': generated_node
                    }
            
            if not is_location_specific:
                # Evaluate text response relevance
                text_scores = []
                num_attempts = 5
                
                relevance_prompt = """Rate the relevance and quality of the response on a scale of 0-100, where:
                0 = Completely irrelevant or incorrect
                100 = Perfect response that fully addresses the query
                
                Consider:
                1. How well the response answers the query
                2. The accuracy and relevance of the information
                3. The completeness of the response
                4. The clarity and usefulness of the explanation
                
                Return only the numerical score."""
                
                for attempt in range(num_attempts):
                    score_prompt = f"""Query: {query}

Response to evaluate:
{generated_response}

Rate the relevance and quality (0-100):"""
                    
                    try:
                        response = await self.chat.llm.generate_response(score_prompt, relevance_prompt)
                        score = float(response.strip())
                        if 0 <= score <= 100:
                            text_scores.append(score)
                    except (ValueError, Exception):
                        continue
                
                avg_text_score = sum(text_scores) / len(text_scores) if text_scores else 0.0
                normalized_score = avg_text_score / 100.0
                
                generation_evaluation = {
                    'semantic_score': normalized_score,
                    'spatial_score': None,
                    'combined_score': normalized_score,
                    'is_location_specific': False
                }

            print("\n=== Generation Evaluation Results ===")
            print(f"Response Type: {'Location-specific' if is_location_specific else 'General text'}")
            if is_location_specific:
                print(f"Average Semantic Score: {avg_semantic_score:.4f}")
                print(f"Spatial Score: {spatial_score:.4f}")
                print(f"Combined Score: {final_score:.4f}")
            else:
                print(f"Text Relevance Score: {normalized_score:.4f}")
            
            return generation_evaluation
            
        except Exception as e:
            print(f"Error evaluating generated response: {str(e)}")
            traceback.print_exc()
            return None

    async def log_experiment(self, query: str, retrieved_nodes: List[Dict], timing_metrics: Dict, 
                           semantic_metrics: Dict, spatial_metrics: Dict) -> None:
        """Log experiment results to a structured file"""
        log_file = self.experiment_log_dir / f"experiment_log_{self.current_experiment_time}.jsonl"
        
        # Extract query location if present
        query_parts = query.split('|')
        location = None
        base_query = query_parts[0].strip()
        for part in query_parts[1:]:
            if 'L:' in part:
                loc = part.split('L:')[1].strip()
                lat, lon = map(float, loc.split(','))
                location = {'latitude': lat, 'longitude': lon}

        # Prepare metrics for top 1
        top1_metrics = {
            'query': base_query,
            'location': location,
            'timestamp': datetime.now().isoformat(),
            'timing': timing_metrics,
            'metrics': {
                'semantic_relativity': semantic_metrics['top1'],
                'std': semantic_metrics['top1_std'],
                'spatial_relativity': spatial_metrics['score'] if retrieved_nodes else 0.0,
            },
            'retrieved_node': retrieved_nodes[0] if retrieved_nodes else None
        }

        # Prepare metrics for top 5
        top5_metrics = {
            'query': base_query,
            'location': location,
            'timestamp': datetime.now().isoformat(),
            'timing': timing_metrics,
            'metrics': {
                'semantic_relativity': semantic_metrics['top5'],
                'std': semantic_metrics['top5_std'],
                'spatial_relativity': spatial_metrics['mean_distance'] if retrieved_nodes else 0.0,
                'comprehensiveness': spatial_metrics['level1_coverage'] if retrieved_nodes else 0.0
            },
            'retrieved_nodes': retrieved_nodes[:5] if retrieved_nodes else []
        }

        # Log to file
        with open(log_file, 'a') as f:
            f.write(json.dumps({'type': 'top1', 'data': top1_metrics}) + '\n')
            f.write(json.dumps({'type': 'top5', 'data': top5_metrics}) + '\n')

        # Print summary
        print("\n=== Experiment Results ===")
        print(f"Query: {base_query}")
        if location:
            print(f"Location: {location}")
        
        print("\nTop 1 Metrics:")
        print(f"├─ Correctness: {top1_metrics['metrics']['semantic_relativity']:.3f}")
        print(f"├─ Affirmness: {top1_metrics['metrics']['std']:.3f}")
        print(f"└─ Spatial Relativity: {top1_metrics['metrics']['spatial_relativity']:.3f}")
        
        print("\nTop 5 Metrics:")
        print(f"├─ Correctness: {top5_metrics['metrics']['semantic_relativity']:.3f}")
        print(f"├─ Affirmness: {top5_metrics['metrics']['std']:.3f}")
        print(f"├─ Spatial Relativity: {top5_metrics['metrics']['spatial_relativity']:.3f}")
        print(f"└─ Comprehensiveness: {top5_metrics['metrics']['comprehensiveness']:.3f}")

    async def evaluate_query(self, query_item: Dict, retrieval_method: str = 'parallel') -> Dict:
        """Evaluate a single query using specified retrieval method"""
        try:
            start_time = time.time()
            query = query_item['query']
            location = query_item['location']
            
            if retrieval_method == 'parallel':
                result = await self.parallel_retriever.retrieve_and_respond(
                    query=query,
                    location=location
                )
                
                retrieved_nodes = result['retrieved_nodes']
                
                # Calculate semantic metrics (returns normalized values 0-1)
                semantic_metrics = await self.compute_semantic_relativity(query, retrieved_nodes)
                
                # Calculate spatial metrics
                spatial_metrics = self.compute_spatial_relativity(location, retrieved_nodes)
                
                # Calculate overall scores (semantic * spatial)
                # Top 1 metrics
                top_1_node = retrieved_nodes[0] if retrieved_nodes else None
                top_1_metrics = {
                    'semantic_relativity': semantic_metrics['top1'],  # Already normalized 0-1
                    'std': semantic_metrics['top1_std'],  # Already normalized 0-1
                    'spatial_relativity': spatial_metrics['top1_spatial'],
                    'overall_score': semantic_metrics['top1'] * spatial_metrics['top1_spatial'],
                    'node_details': {
                        'name': top_1_node['name'],
                        'position': top_1_node['position'],
                        'caption': top_1_node['caption'],
                        'image_path': top_1_node.get('image_path', '')
                    } if top_1_node else None
                }
                
                
                # Top 5 metrics
                top_5_metrics = {
                    'semantic_relativity': semantic_metrics['top5'],
                    'std': semantic_metrics['top5_std'],
                    'spatial_relativity': spatial_metrics['top5_spatial'],
                    'overall_score': semantic_metrics['top5'] * spatial_metrics['top5_spatial']
                }
                
                # Overall query metrics
                overall_metrics = {
                    'comprehensiveness': spatial_metrics['level1_coverage'],
                    'unique_level1_parents': spatial_metrics['unique_level1_parents'],
                    'total_level1_nodes': spatial_metrics['total_level1_nodes'],
                }
                
                timing_metrics = {
                    'total_time': time.time() - start_time,
                    'retrieval_time': result['timing']['retrieval'],
                    'response_time': result['timing']['total'] - result['timing']['retrieval']
                }
                
                return {
                    'query': query,
                    'location': location,
                    'timing': timing_metrics,
                    'top_1': top_1_metrics,
                    'top_5': top_5_metrics,
                    'overall_metrics': overall_metrics,
                    'success': True,
                    'retrieval_method': retrieval_method
                }
                
        except Exception as e:
            print(f"Error in evaluate_query: {str(e)}")
            traceback.print_exc()
            return {
                'error': str(e),
                'success': False,
                'retrieval_method': retrieval_method
            }

    async def run_batch_evaluation(self, query_file: str):
        """Run evaluation on a batch of queries from a file"""
        # Default location for all queries
        default_location = {
            'latitude': 40.443336,
            'longitude': -79.944023
        }
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        batch_results_file = self.experiment_log_dir / f"batch_results_{timestamp}.json"
        
        # Initialize results file with metadata
        batch_metadata = {
            'query_file': query_file,
            'default_location': default_location,
            'total_queries': 0,
            'results': []
        }
        
        with open(batch_results_file, 'w') as f:
            json.dump(batch_metadata, f, indent=2)
        
        # Read queries from file
        with open(query_file, 'r') as f:
            raw_queries = [line.strip() for line in f if line.strip() and not line.startswith('#')]
        
        print(f"\nTotal queries to process: {len(raw_queries)}")
        
        for i, raw_query in enumerate(raw_queries, 1):
            try:
                print(f"\nProcessing query {i}/{len(raw_queries)}: {raw_query}")
                print(f"Using default location: lat={default_location['latitude']}, lon={default_location['longitude']}")
                
                # Format the query with location
                formatted_query = f"L:{default_location['latitude']},{default_location['longitude']} | Q: {raw_query}"
                
                # Create properly formatted query item
                query_item = {
                    'query': raw_query,  # original query without location
                    'location': default_location,
                    'use_history': False,
                    'searchable_term': raw_query.split()[-1],  # Simple extraction of search term
                    'full_query': formatted_query  # Full query with location
                }
                
                # Evaluate query
                result = await self.evaluate_query(query_item, retrieval_method='parallel')
                
                # Update results file
                with open(batch_results_file, 'r') as f:
                    current_data = json.load(f)
                
                current_data['results'].append({
                    'query': raw_query,
                    'formatted_query': formatted_query,
                    'location': default_location,
                    'result': result,
                    'timestamp': datetime.now().isoformat()
                })
                current_data['total_queries'] = len(current_data['results'])
                
                with open(batch_results_file, 'w') as f:
                    json.dump(current_data, f, indent=2)
                
                print(f"Results saved to {batch_results_file}")
                
                # Optional: add delay between queries
                await asyncio.sleep(1)
                
            except Exception as e:
                print(f"Error processing query: {raw_query}")
                print(f"Error: {str(e)}")
                traceback.print_exc()
                
                # Log the error but continue with next query
                error_result = {
                    'query': raw_query,
                    'formatted_query': formatted_query if 'formatted_query' in locals() else None,
                    'location': default_location,
                    'error': str(e),
                    'timestamp': datetime.now().isoformat()
                }
                
                with open(batch_results_file, 'r') as f:
                    current_data = json.load(f)
                
                current_data['results'].append(error_result)
                current_data['total_queries'] = len(current_data['results'])
                
                with open(batch_results_file, 'w') as f:
                    json.dump(current_data, f, indent=2)
        
        # Read final results
        with open(batch_results_file, 'r') as f:
            final_results = json.load(f)
        
        print(f"\nBatch evaluation complete. Processed {final_results['total_queries']} queries.")
        print(f"Results saved to: {batch_results_file}")
        
        return final_results['results']

async def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Run the evaluation system')
    parser.add_argument('--graph-dir', type=str, help='Path to the semantic forest graph file')
    parser.add_argument('--vector-db', type=str, help='Path to the vector database directory')
    parser.add_argument('--image-dir', type=str, help='Path to the images directory')
    parser.add_argument('--mode', type=str, choices=['batch', 'interactive'], 
                       default='interactive', help='Evaluation mode')
    parser.add_argument('--query-file', type=str, help='Path to query file for batch mode')
    
    args = parser.parse_args()
    
    # Initialize evaluator with provided paths
    evaluator = RetrievalEvaluator(
        graph_path=args.graph_dir,
        vector_db_path=args.vector_db,
        image_dir=args.image_dir
    )
    await evaluator.initialize()
    
    if args.mode == 'batch':
        print("\nStarting batch evaluation...")
        # Process both query files
        explicit_queries = '/Users/danielxie/E-RAG/data_generation/explicit_location_queries.txt'
        implicit_queries = '/Users/danielxie/E-RAG/data_generation/Implicit_location_queries.txt'
        
        print("\nProcessing explicit location queries...")
        explicit_results = await evaluator.run_batch_evaluation(explicit_queries)
        
        print("\nProcessing implicit location queries...")
        implicit_results = await evaluator.run_batch_evaluation(implicit_queries)
        
        print("\nEvaluation complete. Results saved to experiment_logs directory.")
    else:  # interactive mode
        while True:
            await evaluator.interactive_mode()
            if input("\nTry another query? [Y/n]: ").lower() == 'n':
                break

if __name__ == "__main__":
    asyncio.run(main()) 