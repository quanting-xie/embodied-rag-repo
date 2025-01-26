from qdrant_client import QdrantClient
from qdrant_client.http import models
from pathlib import Path

# Initialize vector database client
vector_db_path = Path("/Users/danielxie/E-RAG/Embodied-RAG/graph/vector_db")
vector_db = QdrantClient(path=str(vector_db_path))
collection_name = "environmental_nodes"

# Get initial count
initial_count = vector_db.count(collection_name).count  # Extract the count value
print(f"Initial point count: {initial_count}")

# Delete the collection
print("\nDeleting collection...")
vector_db.delete_collection(collection_name)

# Recreate the collection
print("Recreating collection...")
vector_db.create_collection(
    collection_name=collection_name,
    vectors_config=models.VectorParams(
        size=1536,  # OpenAI embedding dimension
        distance=models.Distance.COSINE
    )
)

# Verify deletion
final_count = vector_db.count(collection_name).count  # Extract the count value
print(f"\nCollection recreated: {collection_name}")
print(f"Final point count: {final_count}")
print(f"Points removed: {initial_count - final_count}") 