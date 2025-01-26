from typing import List, Dict, Tuple
import numpy as np
import scipy.stats
import base64
from pathlib import Path
import asyncio
import traceback
import json

class ScoringUtils:
    def __init__(self, llm_interface, image_dir: Path):
        self.llm = llm_interface
        self.image_dir = image_dir

    async def compute_semantic_relativity(self, query: str, retrieved_nodes: List[Dict]) -> Dict[str, float]:
        """Compute semantic relativity using GPT-4 to score relevance on a 0-100 scale in parallel"""
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
                    response = await self.llm.generate_response(
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
            
            return sum(node_scores) / len(node_scores) if node_scores else 0.0

        # Evaluate all nodes in parallel
        print("\nEvaluating nodes in parallel...")
        tasks = [
            get_score_for_node(node, i+1) 
            for i, node in enumerate(retrieved_nodes[:5])  # Only evaluate top 5
        ]
        
        scores_per_node = await asyncio.gather(*tasks)
        
        # Print results
        for i, score in enumerate(scores_per_node, 1):
            print(f"Node {i} Final Average Score: {score:.2f}")

        if not scores_per_node:
            return {'top1': 0.0, 'top5': 0.0}

        # Normalize scores to 0-1 range
        normalized_scores = [score / 100.0 for score in scores_per_node]
        
        return {
            'top1': normalized_scores[0] if normalized_scores else 0.0,
            'top5': sum(normalized_scores) / len(normalized_scores) if normalized_scores else 0.0,
            'raw_scores': scores_per_node  # Include raw scores for reference
        }

    @staticmethod
    def compute_ci(values: List[float], confidence: float = 0.95) -> Tuple[float, float]:
        """Compute confidence interval for a list of values"""
        n = len(values)
        mean = np.mean(values)
        se = np.std(values, ddof=1) / np.sqrt(n)
        h = se * scipy.stats.t.ppf((1 + confidence) / 2, n-1)
        return mean, h

    async def evaluate_generated_response(self, query: str, generated_response: str) -> Dict:
        """Evaluate the generated response using LLM"""
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

            response_type = await self.llm.generate_response(classify_prompt, system_prompt)
            is_location_specific = response_type.strip().lower() == 'location'

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
                
                location_json = await self.llm.generate_response(generated_response, extract_prompt)
                
                # Clean up JSON response
                cleaned_json = location_json.strip()
                if cleaned_json.startswith('```'):
                    cleaned_json = cleaned_json.split('\n', 1)[1]
                if cleaned_json.endswith('```'):
                    cleaned_json = cleaned_json.rsplit('\n', 1)[0]
                cleaned_json = cleaned_json.replace('json\n', '', 1)
                
                try:
                    generated_node = json.loads(cleaned_json)
                    return {
                        'is_location_specific': True,
                        'location_data': generated_node
                    }
                except json.JSONDecodeError:
                    return {
                        'is_location_specific': False,
                        'error': 'Failed to parse location JSON'
                    }
            
            return {
                'is_location_specific': False
            }
            
        except Exception as e:
            return {
                'error': str(e),
                'is_location_specific': False
            }
