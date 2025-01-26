import asyncio
from typing import List, Dict, Any
import networkx as nx
from llm import LLMInterface
from config import Config
import time
import logging
import json
from pathlib import Path
import traceback
from utils.spatial import extract_location_from_query, compute_spatial_score

class ParallelLLMRetriever:
    def __init__(self, graph: nx.Graph, llm_interface: LLMInterface, max_parallel_paths: int = None):
        self.graph = graph
        self.llm = llm_interface
        
        # Get configuration with defaults
        parallel_config = Config.RETRIEVAL.get('parallel_retrieval', {})
        self.k_branches = parallel_config.get('k_branches', 2)
        
        # Setup logging
        self.logger = logging.getLogger('retrieval')
        self.logger.setLevel(logging.INFO)
        if not self.logger.handlers:
            handler = logging.FileHandler('retrieval.log')
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

    async def retrieve_and_respond(self, query: str, location: Dict = None, return_nodes: bool = False) -> Dict:
        """Main method to retrieve context and generate response"""
        start_time = time.time()
        self.logger.info(f"\n=== Starting Retrieval for Query: '{query}' ===")
        
        try:
            # Pass location to retrieve_nodes
            retrieved_nodes = await self.retrieve_nodes(query, location=location)
            retrieval_time = time.time() - start_time
            
            # Build simple context from retrieved nodes
            context = self._build_context(retrieved_nodes)
            
            # Generate response
            response = await self.generate_response(query, context)
            
            if return_nodes:
                return context, retrieved_nodes
            
            return {
                'response': response,
                'retrieved_nodes': retrieved_nodes,
                'context': context,
                'timing': {
                    'retrieval': retrieval_time,
                    'total': time.time() - start_time
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error in retrieve_and_respond: {str(e)}", exc_info=True)
            raise

    async def retrieve_nodes(self, query: str, location: Dict = None, max_context_length: int = 16000) -> List[Dict]:
        """Unified retrieval method that explores k-best nodes and limits by context length"""
        # Use provided location or extract from query
        query_location = location if location else extract_location_from_query(query)
        
        # Get nodes by level
        level_nodes = self._get_nodes_by_level()
        max_level = max(level_nodes.keys())
        scored_results = {}

        async def process_path(start_node: str, current_level: int):
            if current_level < 0:
                return
            
            available_nodes = self._get_available_nodes(level_nodes[current_level], start_node) if start_node else level_nodes[current_level]
            
            if not available_nodes:
                return
            
            nodes_for_selection = [
                {
                    'id': node,
                    'summary': data.get('summary', ''),
                    'name': data.get('name', ''),
                    'type': data.get('type', ''),
                    'level': data.get('level', 0),
                }
                for node, data in available_nodes
            ]
            
            selected_nodes = await self.llm.select_multiple_nodes_with_scores(
                query, 
                nodes_for_selection, 
                k=self.k_branches
            )
            
            # delete duplicate selected nodes
            selected_nodes = list(set(selected_nodes))

            tasks = []
            for node_id, score in selected_nodes:
                if not node_id:
                    continue
                
                node_data = self.graph.nodes[node_id].copy()
                if node_data.get('type') in ['base', 'object']:
                    if 'position' in node_data:
                        pos = node_data['position']
                        if isinstance(pos, dict):
                            node_data['position'] = {
                                'latitude': float(pos.get('y', pos.get('latitude', 0))),
                                'longitude': float(pos.get('x', pos.get('longitude', 0)))
                            }
                    
                    node_data.update({
                        'name': node_data.get('name', ''),
                        'caption': node_data.get('caption', ''),
                        'image_path': node_data.get('image_path', ''),
                        'type': node_data.get('type', 'base'),
                        'score': float(score)
                    })
                    
                    scored_results[node_id] = node_data
                else:
                    tasks.append(process_path(node_id, current_level - 1))
            
            if tasks:
                await asyncio.gather(*tasks)

        await process_path(None, max_level)
        
        semantic_ranked = []
        for node in scored_results.values():
            semantic_score = node['score']
            node['semantic_score'] = semantic_score
            semantic_ranked.append(node)
        
        semantic_ranked.sort(key=lambda x: x['semantic_score'], reverse=True)
        
        semantic_weight = 0.7
        spatial_weight = 0.3
        ranked_nodes = []
        
        for node in scored_results.values():
            semantic_score = node['semantic_score']
            
            if query_location is not None and 'position' in node:
                node_location = {
                    'latitude': float(node['position'].get('y', node['position'].get('latitude', 0))),
                    'longitude': float(node['position'].get('x', node['position'].get('longitude', 0)))
                }
                
                # Use normalized distance-based spatial score
                spatial_score = compute_spatial_score(
                    query_location, 
                    node_location,
                    max_distance=500.0  # 0.5km radius
                )
                node['spatial_score'] = spatial_score
                combined_score = (semantic_weight * semantic_score + spatial_weight * spatial_score)
            else:
                node['spatial_score'] = None
                combined_score = semantic_score
            
            node['combined_score'] = combined_score
            ranked_nodes.append(node)
        
        # Sort by combined score
        ranked_nodes.sort(key=lambda x: x['combined_score'], reverse=True)
      
        # Then filter based on context length
        selected_nodes = []
        current_length = 0
        
        for node in ranked_nodes:
            node_context = self._build_node_context(node)
            node_length = len(node_context)
            
            if current_length + node_length > max_context_length:
                break
            
            selected_nodes.append(node)
            current_length += node_length
        
        print(f"\nFinal selection after context length filtering ({current_length}/{max_context_length}):")
        for i, node in enumerate(selected_nodes, 1):
            score_details = f"combined: {node['combined_score']:.2f}"
            if node['spatial_score'] is not None:
                score_details += f" (semantic: {node['semantic_score']:.2f}, spatial: {node['spatial_score']:.2f})"
            print(f"{i}. {node['name']} - {score_details}")
        
        return selected_nodes

    def _get_nodes_by_level(self) -> Dict[int, List[tuple]]:
        """Get nodes organized by level"""
        print("Entering _get_nodes_by_level")
        level_nodes = {}
        try:
            # Process nodes and explicitly exclude image-related fields
            excluded_fields = {'image_path', 'image', 'image_data', 'image_base64'}
            
            for node, data in self.graph.nodes(data=True):
                level = data.get('level', 0)
                if level not in level_nodes:
                    level_nodes[level] = []
                    
                # Create a clean copy of data without image fields
                filtered_data = {
                    k: v for k, v in data.items() 
                    if k not in excluded_fields
                }
                
                level_nodes[level].append((node, filtered_data))
            
            # Print summary instead of full data
            print(f"\nLevel summary:")
            for level in sorted(level_nodes.keys()):
                node_count = len(level_nodes[level])
                print(f"Level {level}: {node_count} nodes")
                
            return level_nodes
            
        except Exception as e:
            print(f"Error in _get_nodes_by_level: {str(e)}")
            traceback.print_exc()
            return {}

    def _get_available_nodes(self, nodes: List[tuple], parent_node: str = None) -> List[tuple]:
        """Get available nodes for selection, filtered by parent if provided"""
        if parent_node is None:
            return nodes
        
        # Get parent node data
        parent_data = self.graph.nodes[parent_node]
        parent_members = parent_data.get('members', [])
        
        # Ensure parent_members is a list
        if isinstance(parent_members, str):
            parent_members = [parent_members]
        elif not isinstance(parent_members, list):
            parent_members = list(parent_members)
        
        # Filter nodes that are members of the parent node
        return [
            (node, data) for node, data in nodes 
            if (
                # Include node if it's in parent's members list
                node in parent_members or
                str(node) in parent_members or  # Handle string IDs
                # Or if node's ID is in parent's members list
                str(data.get('id', node)) in parent_members
            )
        ]

    def _build_node_context(self, node: Dict) -> str:
        """Build context string for a single node"""
        context_parts = []
        
        context_parts.append(f"\nLocation: {node.get('name', 'unnamed')}")
        if 'caption' in node:
            context_parts.append(f"caption: {node['caption']}")
        if 'position' in node:
            context_parts.append(f"position: {self._format_position(node['position'])}")
        if 'image_path' in node:
            context_parts.append(f"image_path: {node['image_path']}")
        
        return '\n'.join(context_parts)

    def _build_context(self, nodes: List[Dict]) -> str:
        """Build context from selected nodes"""
        if not nodes:
            return "No relevant nodes found."
        
        context_parts = ["=== Retrieved Locations ==="]
        
        for node in nodes:
            context_parts.append(self._build_node_context(node))
        
        return '\n'.join(context_parts)

    async def generate_response(self, query: str, context: str) -> str:
        """Generate response using retrieved context"""
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

        try:
            return await self.llm.generate_response(prompt)
        except Exception as e:
            self.logger.error(f"Error generating response: {str(e)}", exc_info=True)
            return f"Error generating response: {str(e)}"

    def extract_target_position(self, response: str) -> Dict:
        """Extract target position from response"""
        import re
        target_match = re.search(r'<<(.+?)>>', response)
        if target_match:
            target = target_match.group(1)
            for node, data in self.graph.nodes(data=True):
                if (data.get('name') == target or node == target) and 'position' in data:
                    return data['position']
        return None

    def _format_position(self, pos: Any) -> str:
        """Format position data consistently"""
        if isinstance(pos, (list, tuple)) and len(pos) == 3:
            return f"[{pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}]"
        elif isinstance(pos, dict) and all(k in pos for k in ['x', 'y', 'z']):
            return f"[{pos['x']:.2f}, {pos['y']:.2f}, {pos['z']:.2f}]"
        return str(pos)

    def _log_retrieval_stats(self, level_times: Dict[int, float], results: set):
        """Log detailed retrieval statistics"""
        self.logger.info("\n=== Retrieval Statistics ===")
        total_time = sum(level_times.values())
        for level, time_taken in level_times.items():
            self.logger.info(
                f"Level {level}: {time_taken:.2f}s ({(time_taken/total_time)*100:.1f}%)"
            )
        self.logger.info(f"Retrieved Nodes: {len(results)}")
        for node in results:
            self.logger.info(f"- {node}") 

    def get_retrieved_node_data(self) -> List[Dict]:
        """Get the full data for retrieved nodes"""
        return getattr(self, 'retrieved_node_data', []) 