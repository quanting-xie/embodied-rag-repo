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

class ParallelLLMRetriever:
    def __init__(self, graph: nx.Graph, llm_interface: LLMInterface, max_parallel_paths: int = None):
        self.graph = graph
        self.llm = llm_interface
        
        # Get configuration with defaults
        parallel_config = Config.RETRIEVAL.get('parallel_retrieval', {})
        self.max_parallel_paths = (max_parallel_paths or 
                                 parallel_config.get('max_parallel_paths', 3))
        self.k_branches = parallel_config.get('k_branches', 2)
        self.temperature = parallel_config.get('temperature', 0.7)
        
        # Setup logging
        self.logger = logging.getLogger('retrieval')
        self.logger.setLevel(logging.INFO)
        if not self.logger.handlers:
            handler = logging.FileHandler('retrieval.log')
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            
        print(f"Initialized ParallelLLMRetriever with {self.max_parallel_paths} parallel paths")

    async def classify_query_type(self, query: str) -> str:
        """Classify query as either 'find' or 'global'"""
        try:
            self.logger.info(f"Classifying query: {query}")
            
            system_prompt = """You are a query classifier. Classify the query into one of two types:
            1. 'find' - for queries asking to find specific locations or comparing locations
            2. 'global' - for queries about general vibes, patterns, or area-wide information
            Respond with only 'find' or 'global'."""
            
            # Use generate_response with explicit None for image_base64
            response = await self.llm.generate_response(query, system_prompt, image_base64=None)
            query_type = response.strip().lower()
            # Validate response
            if query_type not in ['find', 'global']:
                self.logger.warning(f"Invalid query type '{query_type}', defaulting to 'find'")
                query_type = 'find'
                
            self.logger.info(f"Query '{query}' classified as: {query_type}")
            return query_type
            
        except Exception as e:
            self.logger.error(f"Error in query classification: {str(e)}", exc_info=True)
            # Default to 'find' on error
            return 'find'

    async def retrieve_and_respond(self, query: str) -> Dict:
        """Main method to retrieve context and generate response"""
        start_time = time.time()
        self.logger.info(f"\n=== Starting Retrieval for Query: '{query}' ===")
        
        try:
            # 1. Classify Query Type
            query_type = await self.classify_query_type(query)
            self.logger.info(f"Query classified as: {query_type}")
            
            # 2. Parallel Retrieval based on query type
            if query_type == 'find':
                retrieved_nodes = await self.parallel_find_retrieve(query)
            else:  # global
                retrieved_nodes = await self.parallel_global_retrieve(query)
            
            self.logger.info(f"Retrieved {len(retrieved_nodes)} nodes")
            retrieval_time = time.time() - start_time
            
            # 3. Build Context from node dictionaries directly
            context = self._build_hierarchical_context(retrieved_nodes)
            
            # 4. Generate Response
            response = await self.generate_response(query, context, query_type)
            
            # 5. Extract Navigation Target
            target_position = self.extract_target_position(response)
            
            return {
                'response': response,
                'target_position': target_position,
                'retrieved_nodes': retrieved_nodes,  # Now contains full node data
                'context': context,
                'timing': {
                    'retrieval': retrieval_time,
                    'total': time.time() - start_time
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error in retrieve_and_respond: {str(e)}", exc_info=True)
            raise

    async def parallel_find_retrieve(self, query: str) -> List[Dict]:
        """Parallel retrieval for 'find' queries - each path selects best single node"""
        unique_results = {}  # Use dict to track unique nodes by ID/name
        
        try:
            print("entering parallel_find_retrieve")
            level_nodes = self._get_nodes_by_level()
            max_level = max(level_nodes.keys())
            
            paths = [{'chain': [], 'current_level': max_level} 
                    for _ in range(self.max_parallel_paths)]

            async def process_find_path(path_info, path_num):
                try:
                    print(f"\nProcessing path {path_num}:")
                    chain = path_info['chain']
                    current_level = path_info['current_level']
                    results = []
                    
                    while current_level >= 0:
                        print(f"  Level {current_level}:")
                        available_nodes = self._get_available_nodes(
                            level_nodes[current_level], 
                            chain[-1] if chain else None
                        )
                        
                        if not available_nodes:
                            print("    No available nodes, breaking")
                            break
                            
                        selected_node = await self.llm.select_best_node(
                            query, 
                            available_nodes, 
                            mode='find'
                        )
                        
                        if not selected_node:
                            print("    No node selected, breaking")
                            break
                            
                        chain.append(selected_node)
                        node_type = self.graph.nodes[selected_node].get('type')
                        
                        if node_type in ['base', 'object']:
                            node_data = self.graph.nodes[selected_node].copy()
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
                                'score': 1.0
                            })
                            
                            # Use node ID or name as unique identifier
                            node_key = selected_node or node_data.get('name', '')
                            results.append((node_key, node_data))
                            print(f"    Found result node: {node_data['name']} at position: {node_data.get('position')}")
                            break
                        
                        current_level -= 1
                        
                    return results
                    
                except Exception as e:
                    print(f"Error in process_find_path {path_num}: {str(e)}")
                    traceback.print_exc()
                    return []

            # Run paths in parallel
            path_results = await asyncio.gather(*[
                process_find_path(path, i) for i, path in enumerate(paths)
            ])
            
            # Combine results, maintaining uniqueness
            for results in path_results:
                for node_key, node_data in results:
                    if node_key not in unique_results:
                        unique_results[node_key] = node_data
            
            # Convert back to list
            expanded_results = list(unique_results.values())
            
            print(f"\nFound {len(expanded_results)} unique results with position data:")
            for node in expanded_results:
                print(f"- Node: {node.get('name', 'unnamed')} at position: {node.get('position')}")
            
            # Store for visualization
            self.retrieved_node_data = expanded_results
            
            return expanded_results

        except Exception as e:
            print(f"Error in parallel_find_retrieve: {str(e)}")
            traceback.print_exc()
            return []

    async def parallel_global_retrieve(self, query: str) -> List[str]:
        """Parallel retrieval for 'global' queries - gather k best nodes at each level"""
        expanded_results = set()
        level_nodes = self._get_nodes_by_level()
        
        async def process_level(level: int, parent_nodes: List[str] = None):
            if level < 0:
                return []
                
            available_nodes = []
            if parent_nodes:
                for parent in parent_nodes:
                    available_nodes.extend(self._get_available_nodes(level_nodes[level], parent))
            else:
                available_nodes = level_nodes[level]
            
            if not available_nodes:
                return []
            
            # Prepare nodes for selection
            nodes_for_selection = [
                {
                    'id': node,
                    'summary': data.get('summary', 'No summary'),
                    'level': data.get('level', 0),
                    'type': data.get('type', 'unknown'),
                    'name': data.get('name', node)
                }
                for node, data in available_nodes
            ]
            
            # Select k best nodes at this level
            selected_nodes = await self.llm.select_multiple_nodes(
                query, nodes_for_selection, k=self.k_branches, mode='global'
            )
            
            # Recursively process next level
            child_nodes = await process_level(level - 1, selected_nodes)
            return selected_nodes + child_nodes
        
        # Start from top level
        max_level = max(level_nodes.keys())
        all_nodes = await process_level(max_level)
        return list(set(all_nodes))


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

    def _build_hierarchical_context(self, nodes: List[Dict]) -> str:
        """Build hierarchical context from node dictionaries"""
        if not nodes:
            return "No relevant nodes found."
            
        context = ["=== Retrieved Locations ==="]
        
        for node in nodes:
            context.append(f"\nLocation: {node.get('name', 'unnamed')}")
            if 'caption' in node:
                context.append(f"Description: {node['caption']}")
            if 'position' in node:
                context.append(f"Position: {self._format_position(node['position'])}")
            if 'type' in node:
                context.append(f"Type: {node['type']}")
            
        return "\n".join(context)

    async def generate_response(self, query: str, context: str, query_type: str) -> str:
        """Generate response using retrieved context"""
        try:
            return await self.llm.generate_response(query, context, query_type)
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