import networkx as nx
from pathlib import Path
import asyncio
from tqdm import tqdm
from qdrant_client import QdrantClient
from qdrant_client.http import models
import uuid
from llm import LLMInterface
import argparse

async def index_existing_forest(forest_path: Path, collection_name: str = "environmental_nodes", batch_size: int = 100):
    """Index an existing semantic forest into the vector database with parallel processing"""
    print(f"\nIndexing forest from: {forest_path}")
    
    # Initialize LLM interface for embeddings
    llm_interface = LLMInterface()
    
    # Setup vector database
    vector_db_path = forest_path.parent.parent.parent / "vector_db"
    print(f"Using vector database at: {vector_db_path}")
    vector_db = QdrantClient(path=str(vector_db_path))
    
    # Ensure collection exists
    try:
        collections = vector_db.get_collections()
        collection_exists = any(c.name == collection_name for c in collections.collections)
        
        if not collection_exists:
            print(f"Creating collection: {collection_name}")
            vector_db.create_collection(
                collection_name=collection_name,
                vectors_config=models.VectorParams(
                    size=1536,
                    distance=models.Distance.COSINE
                )
            )
    except Exception as e:
        print(f"Error setting up collection: {str(e)}")
        raise
    
    # Load the forest
    print("Loading semantic forest...")
    G = nx.read_gml(str(forest_path))
    total_nodes = len(G.nodes())
    print(f"Found {total_nodes} nodes to index")
    
    # Prepare all node texts first
    print("\nPreparing node texts...")
    node_texts = []
    node_ids = []
    node_data_list = []
    
    for node_id, node_data in G.nodes(data=True):
        node_text = f"""
        Name: {node_data.get('name', '')}
        Description: {node_data.get('caption', '')}
        Type: {node_data.get('type', 'base')}
        Level: {node_data.get('level', 0)}
        Position: {node_data.get('position', {})}
        Summary: {node_data.get('summary', '')}
        Relationships: {', '.join(node_data.get('relationships', []))}
        """
        node_texts.append(node_text)
        node_ids.append(node_id)
        node_data_list.append(node_data)
    
    # Process embeddings in parallel batches
    print("\nGenerating embeddings in parallel batches...")
    failed_nodes = []
    total_batches = (len(node_texts) + batch_size - 1) // batch_size
    
    for batch_idx in tqdm(range(total_batches)):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, len(node_texts))
        
        # Get current batch
        batch_texts = node_texts[start_idx:end_idx]
        batch_ids = node_ids[start_idx:end_idx]
        batch_data = node_data_list[start_idx:end_idx]
        
        try:
            # Generate embeddings for batch in parallel
            embedding_response = await llm_interface.client.embeddings.create(
                model="text-embedding-ada-002",
                input=batch_texts
            )
            embeddings = [e.embedding for e in embedding_response.data]
            
            # Create points for batch
            points = []
            for i, embedding in enumerate(embeddings):
                point = models.PointStruct(
                    id=str(uuid.uuid4()),
                    vector=embedding,
                    payload={
                        "original_id": batch_ids[i],
                        "node_data": batch_data[i],
                        "text": batch_texts[i],
                        "level": batch_data[i].get('level', 0)
                    }
                )
                points.append(point)
            
            # Upload batch
            vector_db.upsert(
                collection_name=collection_name,
                points=points
            )
            
            await asyncio.sleep(0.1)  # Rate limiting between batches
            
        except Exception as e:
            print(f"\nError processing batch {batch_idx + 1}: {str(e)}")
            failed_nodes.extend([(node_id, str(e)) for node_id in batch_ids])
            continue
    
    # Print results
    total_indexed = vector_db.count(collection_name).count
    print(f"\nIndexing complete:")
    print(f"- Total nodes indexed: {total_indexed}")
    print(f"- Failed to index: {len(failed_nodes)} nodes")
    
    if failed_nodes:
        print("\nFailed nodes:")
        for node_id, error in failed_nodes[:10]:
            print(f"- Node {node_id}: {error}")
        if len(failed_nodes) > 10:
            print(f"... and {len(failed_nodes) - 10} more")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Index existing semantic forest into vector database')
    parser.add_argument('--forest-path', type=str, 
                       default="/Users/danielxie/E-RAG/Embodied-RAG/graph/semantic_forests/graph/semantic_forest_graph.gml",
                       help='Path to semantic forest GML file')
    parser.add_argument('--collection', type=str, 
                       default="environmental_nodes",
                       help='Name of the vector database collection')
    parser.add_argument('--batch-size', type=int,
                       default=100,
                       help='Number of nodes to process in parallel')
    
    args = parser.parse_args()
    
    forest_path = Path(args.forest_path)
    if not forest_path.exists():
        print(f"Error: Forest file not found at {forest_path}")
        exit(1)
        
    asyncio.run(index_existing_forest(forest_path, args.collection, args.batch_size)) 