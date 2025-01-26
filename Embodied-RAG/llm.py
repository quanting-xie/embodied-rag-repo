from openai import AsyncOpenAI
from config import Config
import re
import os
import traceback
import asyncio
import numpy as np
import json
from pathlib import Path
from typing import List
from tqdm import tqdm
import uuid

class LLMInterface:
    def __init__(self):
        # Get OpenAI API key from environment variable
        self.api_key = os.getenv('OPENAI_API_KEY')
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY environment variable is not set")
        
        self.model = Config.LLM['model']
        self.temperature = Config.LLM['temperature']
        self.max_tokens = Config.LLM['max_tokens']
        self.max_retries = 3
        
        print("Using standard OpenAI configuration")
        self.client = AsyncOpenAI(
            api_key=self.api_key,
            base_url="https://api.openai.com/v1" 
        )
        
        print(f"LLM Interface initialized with:")
        print(f"- Model: {self.model}")
        print(f"- API Key present: {bool(self.api_key)}")
        print(f"- Base URL: {self.client.base_url}")


    async def generate_response(self, prompt: str, system_prompt: str = None, image_base64: str = None) -> str:
        """Base method for generating responses from the LLM"""
        if system_prompt is None:
            system_prompt = ""
            
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        if image_base64:
            messages.append({
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_base64}"
                        }
                    }
                ]
            })
        else:
            messages.append({"role": "user", "content": prompt})

        for attempt in range(self.max_retries):
            try:
                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens
                )
                return response.choices[0].message.content.strip()
            except Exception as e:
                print(f"Attempt {attempt + 1} failed: {str(e)}")
                if attempt == self.max_retries - 1:
                    raise
                await asyncio.sleep(1 * (attempt + 1))

    async def generate_text_response(self, prompt: str, system_prompt: str = None) -> str:
        """Text-only response generation - no image processing"""
        return await self.generate_response(prompt, system_prompt, image_base64=None)

    async def generate_embeddings(self, texts, batch_size=100):
        """Generate embeddings for a list of texts in batches"""
        try:
            all_embeddings = []
            
            # Clean all texts first
            cleaned_texts = []
            for text in texts:
                # Convert to string and basic cleaning
                text = str(text).strip()
                
                # Remove markdown-style formatting
                text = text.replace('**NAME:**', 'Name:')
                text = text.replace('**DESCRIPTION:**', 'Description:')
                text = text.replace('**', '')
                
                # Replace newlines with spaces and remove multiple spaces
                text = ' '.join(text.split())
                
                # Handle empty strings
                if not text:
                    text = "empty text"
                
                    
                cleaned_texts.append(text)

            # Process in batches
            print(f"Processing {len(cleaned_texts)} texts in batches of {batch_size}")
            for i in range(0, len(cleaned_texts), batch_size):
                batch = cleaned_texts[i:i + batch_size]
                try:
                    response = await self.client.embeddings.create(
                        model="text-embedding-ada-002",
                        input=batch
                    )
                    all_embeddings.extend([e.embedding for e in response.data])
                    print(f"Processed batch {i//batch_size + 1}/{(len(cleaned_texts) + batch_size - 1)//batch_size}")
                    
                    # Add small delay between batches
                    await asyncio.sleep(0.1)
                    
                except Exception as e:
                    print(f"Error in batch {i//batch_size + 1}: {str(e)}")
                    print(f"Batch size: {len(batch)}")
                    print(f"First text in failed batch: {batch[0][:200]}...")
                    raise

            return np.array(all_embeddings)
            
        except Exception as e:
            print(f"Error generating embeddings: {str(e)}")
            raise

    async def generate_cluster_summary(self, cluster_data):
        """Generate summary for a cluster"""
        system_prompt = """You are a geographic analyst. Your task is to name and describe urban areas based on their visual characteristics."""
        
        prompt = f"""Analyze these captions given, extract the key entities and relationships between them, and then create a summary of the group captions:

Location Descriptions:
{cluster_data}

Return your analysis in this exact JSON format:
{{
    "name": "3-5 word descriptive name",
    "summary": "2-3 sentences describing key features",
    "relationships": ["relationships"]
}}

Example Response:
{{
    "name": "Forested Campus Pathway Area",
    "summary": "A network of pedestrian pathways through a densely wooded campus setting. Large deciduous trees create natural archways over the paths.",
    "relationships": [
        "Paths connect building clusters",
        "Trees line walkways",
        "Green spaces border paths"
    ]
}}"""

        try:
            response = await self.generate_response(prompt, system_prompt)
            if not response:
                raise ValueError("Empty response from LLM")
            
            print(f"\nRaw LLM Response:\n{response}")
            return self.parse_summary_response(response)
            
        except Exception as e:
            print(f"Error in generate_cluster_summary: {str(e)}")
            raise

    async def process_batch_summaries(self, batch_requests):
        """Process a batch of summary requests"""
        batch_input_path = Path("batch_summaries.jsonl")
        try:
            # Save batch requests
            with open(batch_input_path, "w") as f:
                for request in batch_requests:
                    f.write(json.dumps(request) + "\n")
            
            # Create batch file
            batch_file = await self.client.files.create(
                file=open(batch_input_path, "rb"),
                purpose="batch"
            )
            
            # Create and monitor batch job
            batch = await self.client.batches.create(
                input_file_id=batch_file.id,
                endpoint="/v1/chat/completions",
                completion_window="24h"
            )
            
            while True:
                batch_status = await self.client.batches.retrieve(batch.id)
                if batch_status.status in ['completed', 'failed', 'expired']:
                    break
                await asyncio.sleep(30)
            
            if batch_status.status == 'completed':
                output_file = await self.client.files.content(batch_status.output_file_id)
                return [json.loads(line) for line in output_file.text.split('\n') if line]
            
            return None
            
        finally:
            if batch_input_path.exists():
                batch_input_path.unlink()

    def parse_summary_response(self, content):
        """Parse JSON summary response"""
        try:
            # First remove any markdown code block formatting
            content = re.sub(r'```json\s*', '', content)
            content = re.sub(r'```\s*', '', content)
            
            # Find and parse the JSON object
            json_match = re.search(r'\{[\s\S]*\}', content)
            if not json_match:
                print("No JSON object found in response")
                raise ValueError("Invalid response format")
            
            json_str = json_match.group(0)
            data = json.loads(json_str)
            
            # Validate required fields exist
            required_fields = ['name', 'summary', 'relationships']
            missing_fields = [field for field in required_fields if field not in data]
            if missing_fields:
                print(f"Missing required fields: {missing_fields}")
                raise ValueError("Missing required fields")
            
            # Basic validation
            if not isinstance(data['name'], str) or not data['name'].strip():
                raise ValueError("Name cannot be empty")
            
            if not isinstance(data['summary'], str) or not data['summary'].strip():
                raise ValueError("Summary cannot be empty")
            
            if not isinstance(data['relationships'], list) or not data['relationships']:
                raise ValueError("Relationships must be a non-empty list")
            
            # Clean and return the data
            return {
                'name': data['name'].strip(),
                'summary': data['summary'].strip(),
                'relationships': [r.strip() for r in data['relationships'] if r.strip()]
            }
            
        except json.JSONDecodeError as e:
            print(f"JSON parsing error: {str(e)}")
            print(f"Content received: {content[:200]}...")
            raise
            
        except Exception as e:
            print(f"Error parsing response: {str(e)}")
            print(f"Content received: {content[:200]}...")
            raise

    async def batch_generate_summaries(self, cluster_texts):
        """Generate summaries for multiple clusters"""
        summaries = []
        
        try:
            # Process each text individually
            for text in tqdm(cluster_texts, desc="Generating summaries"):
                try:
                    response = await self.client.chat.completions.create(
                        model=self.model,
                        messages=[
                            {"role": "system", "content": "You are a geographic analyst. Return your analysis in JSON format."},
                            {"role": "user", "content": f"""Analyze these location descriptions and create a descriptive summary:

Location Descriptions:
{text}

Return your analysis as JSON:
{{
    "name": "descriptive name",
    "summary": "2-3 sentences describing key features",
    "relationships": ["relationship 1", "relationship 2", "relationship 3"]
}}"""}
                        ],
                        temperature=self.temperature,
                        max_tokens=self.max_tokens
                    )
                    
                    content = response.choices[0].message.content
                    summary = self.parse_summary_response(content)
                    summaries.append(summary)
                    
                    await asyncio.sleep(0.1)
                    
                except Exception as e:
                    import traceback
                    print(f"Error generating summary: {str(e)}")
                    print(traceback.format_exc())
                    raise
            
            return summaries
            
        except Exception as e:
            import traceback
            print(f"Error processing batch: {str(e)}")
            print(traceback.format_exc())
            raise

    async def batch_generate_embeddings(self, texts: List[str], batch_size: int = 100) -> List[np.ndarray]:
        """
        Generate embeddings for a large batch of texts efficiently
        
        Args:
            texts: List of texts to generate embeddings for
            batch_size: Number of texts to process in each batch
            
        Returns:
            List of embeddings as numpy arrays
        """
        try:
            all_embeddings = []
            total_batches = (len(texts) + batch_size - 1) // batch_size
            
            print(f"\nGenerating embeddings for {len(texts)} texts in {total_batches} batches")
            
            # Process in batches
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                
                # Clean batch texts
                cleaned_batch = []
                for text in batch:
                    # Convert to string and basic cleaning
                    text = str(text).strip()
                    
                    # Remove markdown-style formatting
                    text = text.replace('**NAME:**', 'Name:')
                    text = text.replace('**DESCRIPTION:**', 'Description:')
                    text = text.replace('**', '')
                    
                    # Replace newlines with spaces and remove multiple spaces
                    text = ' '.join(text.split())
                    
                    # Handle empty strings
                    if not text:
                        text = "empty text"
                        
                    cleaned_batch.append(text)
                
                try:
                    # Generate embeddings for batch
                    response = await self.client.embeddings.create(
                        model="text-embedding-ada-002",
                        input=cleaned_batch
                    )
                    
                    # Extract embeddings from response
                    batch_embeddings = [e.embedding for e in response.data]
                    all_embeddings.extend(batch_embeddings)
                    
                    print(f"✓ Processed batch {i//batch_size + 1}/{total_batches}")
                    
                    # Add small delay between batches
                    await asyncio.sleep(0.1)
                    
                except Exception as e:
                    print(f"Error in batch {i//batch_size + 1}: {str(e)}")
                    print(f"First text in failed batch: {cleaned_batch[0][:200]}...")
                    # Return empty embeddings for failed texts
                    all_embeddings.extend([np.zeros(1536) for _ in range(len(batch))])
            
            return all_embeddings
            
        except Exception as e:
            print(f"Error in batch_generate_embeddings: {str(e)}")
            # Return empty embeddings for all texts
            return [np.zeros(1536) for _ in range(len(texts))]

    async def select_best_node(self, query: str, nodes: List[tuple], mode: str = 'find') -> str:
        """
        Select the best node from a list of nodes based on the query.
        
        Args:
            query: User's query
            nodes: List of tuples (node_id, node_data)
            mode: 'find' or 'global' mode
        Returns:
            Selected node ID
        """
        if not nodes:
            return None
        
        # Format nodes for LLM consumption
        nodes_for_selection = []
        for node in nodes:
            # Handle both tuple and dict formats
            if isinstance(node, tuple) and len(node) == 2:
                node_id, data = node
            else:
                # If not a tuple, assume it's a dict-like object
                node_id = node.get('id') or str(node)
                data = node
            
            node_info = {
                'id': str(node_id),  # Ensure ID is string
                'name': data.get('name', str(node_id)),
                'level': data.get('level', 0),
                'type': data.get('type', 'unknown'),
                'summary': data.get('summary', ''),
                'members': data.get('members', []),
                'label': data.get('label', '')
            }
            nodes_for_selection.append(node_info)

        system_prompt = f"""You are a location selector. For a {mode} query, select the single most relevant location that best matches the query.
        Consider:
        1. The semantic relevance to the query
        2. The level in the hierarchy (higher levels contain member nodes)
        3. The type of node (cluster or object)
        4. The summary and relationships described
        
        Return only the ID of the best matching node."""
        
        prompt = f"""Query: {query}

Available Locations:
{json.dumps(nodes_for_selection, indent=2)}

Select the single best matching location ID."""
        
        try:
            response = await self.generate_response(prompt, system_prompt, image_base64=None)
            # Clean up response to ensure we get a valid node ID
            node_id = response.strip().split('\n')[0].strip()
            # Verify the selected node exists in our input
            if any(n['id'] == node_id for n in nodes_for_selection):
                return node_id
            # Fallback to first node if selection is invalid
            return nodes[0][0] if isinstance(nodes[0], tuple) else nodes[0].get('id')
        except Exception as e:
            print(f"Error in select_best_node: {str(e)}")
            # Fallback to first node
            return nodes[0][0] if isinstance(nodes[0], tuple) else nodes[0].get('id')

    async def select_multiple_nodes(self, query: str, nodes: List[dict], k: int) -> List[str]:
        """
        Select k best nodes using LLM.
        
        Args:
            query: User's query
            nodes: List of node dictionaries
            k: Number of nodes to select
        Returns:
            List of selected node IDs
        """
        # Format nodes for selection
        nodes_for_selection = []
        for node in nodes:
            # Handle both dictionary formats
            node_id = node.get('id', '')
            if not node_id:  # If id not in dict, the id might be the first element
                node_id = str(node[0]) if isinstance(node, tuple) else ''
            
            node_info = {
                'id': node_id,
                'name': node.get('name', 'Unnamed'),
                'summary': node.get('summary', ''),
                'type': node.get('type', 'unknown'),
                'level': node.get('level', 0),
            }
            nodes_for_selection.append(node_info)

        system_prompt = (
            f"You are a location selector. Your task is to select exactly {k} most relevant locations "
            f"that best match the query. You must return ONLY the node IDs in a comma-separated format."
        )
        
        prompt = (
            f"Query: {query}\n\n"
            f"Available Locations:\n"
            f"{json.dumps(nodes_for_selection, indent=2)}\n\n"
            f"Instructions:\n"
            f"1. Select exactly {k} locations that best match the query\n"
            f"2. Return ONLY their IDs in a comma-separated format\n"
            f"3. Do not include any other text or explanations\n\n"
            f"Example response format:\n"
            f"id1,id2,id3\n\n"
            f"Your response:"
        )
        
        try:
            response = await self.generate_response(prompt, system_prompt)
            # Clean and validate response
            selected_ids = [id.strip() for id in response.strip().split(',') if id.strip()]
            
            # Validate selected IDs exist in our nodes
            valid_ids = [
                node_id for node_id in selected_ids 
                if any(n['id'] == node_id for n in nodes_for_selection)
            ]
            
            # If we don't have enough valid IDs, pad with available IDs
            if len(valid_ids) < k and nodes_for_selection:
                available_ids = [n['id'] for n in nodes_for_selection]
                valid_ids.extend(available_ids[:k - len(valid_ids)])
            
            return valid_ids[:k]
        except Exception as e:
            print(f"Error in select_multiple_nodes: {str(e)}")
            # Fallback: return first k available node IDs
            return [n['id'] for n in nodes_for_selection[:k]] if nodes_for_selection else []

    async def select_multiple_nodes_with_scores(self, query: str, nodes: List[dict], k: int) -> List[tuple]:
        """
        Select k best nodes using LLM and assign relevance scores.
        Returns list of (node_id, score) tuples.
        """
        nodes_for_selection = []
        for node in nodes:
            node_id = node.get('id', '')
            if not node_id:
                node_id = str(node[0]) if isinstance(node, tuple) else ''
            
            node_info = {
                'id': node_id,
                'name': node.get('name', 'Unnamed'),
                'summary': node.get('summary', ''),
                'type': node.get('type', 'unknown'),
                'level': node.get('level', 0),
            }
            nodes_for_selection.append(node_info)

        system_prompt = (
            "You are a location selector analyzing locations to find the most relevant ones for a given query. "
            "You must return your selections in a specific JSON format."
        )
        
        prompt = (
            f"Query: {query}\n\n"
            f"Available Locations:\n"
            "[\n"
        )
        
        # Add each location with clear formatting
        for node in nodes_for_selection:
            prompt += (
                f"  {{\n"
                f"    'id': '{node['id']}',\n"
                f"    'name': '{node['name']}',\n"
                f"    'summary': '{node['summary']}'\n"
                f"  }},\n"
            )
        
        prompt += (
            "]\n\n"
            f"Instructions:\n"
            f"1. Select the {k} most relevant locations for the query\n"
            f"2. For each selected location, assign a relevance score between the query and the location (0-100)\n"
            f"3. Return your response in EXACTLY this JSON format:\n"
            f"[\n"
            f"  {{'id': 'node_id1', 'score': 95}},\n"
            f"  {{'id': 'node_id2', 'score': 80}}\n"
            f"]\n\n"
            f"Your response (ONLY the JSON array):\n"
        )
        
        try:
            response = await self.generate_response(prompt, system_prompt)
            
            # Clean up the response to handle common formatting issues
            cleaned_response = response.strip()
            if not cleaned_response.startswith('['):
                # Try to find the JSON array in the response
                import re
                match = re.search(r'\[(.*?)\]', cleaned_response, re.DOTALL)
                if match:
                    cleaned_response = match.group(0)
                else:
                    raise ValueError("No JSON array found in response")
            
            # Replace single quotes with double quotes for valid JSON
            cleaned_response = cleaned_response.replace("'", '"')
            
            # Parse the JSON
            selected = json.loads(cleaned_response)
            
            # Validate and format results
            valid_selections = []
            for item in selected:
                node_id = item.get('id')
                score = item.get('score', 0)
                if node_id and any(n['id'] == node_id for n in nodes_for_selection):
                    valid_selections.append((node_id, min(max(float(score), 0), 100)))
            
            # If we don't have enough valid selections, add more with low scores
            if len(valid_selections) < k:
                remaining_nodes = [
                    n['id'] for n in nodes_for_selection 
                    if n['id'] not in [x[0] for x in valid_selections]
                ]
                valid_selections.extend(
                    [(node_id, 0.0) for node_id in remaining_nodes[:k-len(valid_selections)]]
                )
            
            print(f"Selected nodes with scores: {valid_selections}")
            return valid_selections[:k]
            
        except Exception as e:
            print(f"Error in select_multiple_nodes_with_scores: {str(e)}")
            print(f"Raw response: {response}")
            # Fallback: return first k nodes with zero scores
            return [(n['id'], 0.0) for n in nodes_for_selection[:k]]
