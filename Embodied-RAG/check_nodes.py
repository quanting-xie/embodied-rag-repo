import networkx as nx
from pathlib import Path

# Load the graph
graph_path = "/Users/danielxie/E-RAG/Embodied-RAG/graph/semantic_forests/graph/semantic_forest_graph.gml"
G = nx.read_gml(graph_path)

# Count total nodes
total_nodes = len(G.nodes())

# Count nodes by level
level_counts = {}
for node, data in G.nodes(data=True):
    level = data.get('level', 0)
    level_counts[level] = level_counts.get(level, 0) + 1

print(f"\nTotal nodes in graph: {total_nodes}")
print("\nNodes by level:")
for level, count in sorted(level_counts.items()):
    print(f"Level {level}: {count} nodes") 