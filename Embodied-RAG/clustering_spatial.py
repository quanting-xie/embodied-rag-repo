import numpy as np
import networkx as nx
from sklearn.cluster import AgglomerativeClustering
import re
from config import Config
import traceback
from math import radians, sin, cos, sqrt, atan2

class SemanticClusterer:
    def __init__(self, llm_interface):
        self.llm = llm_interface
        self.base_threshold = Config.SPATIAL['cluster_distance_threshold']
        self.level_multiplier = Config.SPATIAL['level_multiplier']
        self.max_levels = Config.RETRIEVAL['max_hierarchical_level']
        self.semantic_weight = 0.8
        self.G = None

    def haversine_distance(self, lat1, lon1, lat2, lon2):
        """Calculate the great circle distance between two points in meters"""
        R = 6371000  # Earth radius in meters

        # Convert decimal degrees to radians
        lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])

        # Haversine formula
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        c = 2 * atan2(sqrt(a), sqrt(1-a))
        distance = R * c

        return distance

    async def _compute_similarity_matrix(self, nodes, positions):
        """Compute similarity matrix using both spatial and semantic features"""
        n_nodes = len(nodes)
        similarity_matrix = np.zeros((n_nodes, n_nodes))
        
        # Get node texts for embedding
        node_texts = []
        for node_id in nodes:
            node_data = self.G.nodes[node_id]
            if node_data.get('type') == 'cluster':
                text = f"""
                Name: {node_data.get('name', '')}
                """
                # Summary: {node_data.get('summary', '')}
                # Relationships: {', '.join(node_data.get('relationships', []))}
                # """
            else:
                text = node_data.get('caption', 'unnamed location')
            node_texts.append(text)
        
        # Get embeddings using LLM interface
        text_embeddings = await self.llm.generate_embeddings(node_texts)
        
        # Compute similarities
        for i in range(n_nodes):
            for j in range(i + 1, n_nodes):
                spatial_dist = np.linalg.norm(positions[i] - positions[j])
                spatial_sim = 1 / (1 + spatial_dist)
                
                semantic_sim = np.dot(text_embeddings[i], text_embeddings[j]) / (
                    np.linalg.norm(text_embeddings[i]) * np.linalg.norm(text_embeddings[j]))
                
                combined_sim = (1 - self.semantic_weight) * spatial_sim + self.semantic_weight * semantic_sim
                
                similarity_matrix[i, j] = combined_sim
                similarity_matrix[j, i] = combined_sim
        
        return similarity_matrix

    async def cluster_nodes(self, nodes, positions, level):
        """Perform hierarchical clustering on nodes with geographic distance"""
        if len(nodes) < 2:
            return []
            
        threshold = self.base_threshold * (self.level_multiplier ** (level - 1))
        print(f"\nLevel {level} clustering:")
        print(f"- Distance threshold: {threshold:.2f} meters")
        print(f"- Number of nodes: {len(nodes)}")
        
        # Compute geographic distances between all pairs
        n_nodes = len(nodes)
        distance_matrix = np.zeros((n_nodes, n_nodes))
        distances = []
        
        for i in range(n_nodes):
            for j in range(i + 1, n_nodes):
                # Note: positions has [longitude, latitude]
                lon1, lat1 = positions[i]
                lon2, lat2 = positions[j]
                
                # Calculate actual geographic distance in meters
                dist = self.haversine_distance(lat1, lon1, lat2, lon2)
                distance_matrix[i, j] = dist
                distance_matrix[j, i] = dist
                distances.append(dist)
        
        if distances:
            print(f"- Min distance between points: {min(distances):.2f} meters")
            print(f"- Max distance between points: {max(distances):.2f} meters")
            print(f"- Mean distance between points: {np.mean(distances):.2f} meters")
        
        # Perform clustering with the distance matrix
        clustering = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=threshold,
            metric='precomputed',
            linkage='complete'
        )
        
        labels = clustering.fit_predict(distance_matrix)
        
        # Group nodes by cluster
        clusters = {}
        for i, label in enumerate(labels):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append((nodes[i], positions[i]))
        
        print(f"- Created {len(clusters)} clusters")
        sizes = [len(c) for c in clusters.values()]
        if sizes:
            print(f"- Cluster sizes: min={min(sizes)}, max={max(sizes)}, mean={np.mean(sizes):.1f}")
        
        return clusters

    async def extract_relationships(self, objects):
        """Enhanced hierarchical clustering with semantic feedback"""
        self.G = nx.Graph()
        
        # Create base nodes
        for obj in objects:
            self.G.add_node(
                obj['id'],
                type='location',
                position=obj['position'],
                name=obj.get('name', ''),
                caption=obj.get('caption', ''),
                timestamp=obj.get('timestamp', ''),
                level=0
            )
        
        # Iterative clustering
        current_level = 1
        while current_level <= self.max_levels:
            print(f"\nProcessing Level {current_level}...")
            clusters = await self._process_level(current_level)
            if not clusters:
                break
            current_level += 1
        
        return self.G

    async def _process_level(self, level):
        """Process a single level of clustering"""
        nodes_to_cluster = [
            node for node, data in self.G.nodes(data=True)
            if data.get('level', 0) == level - 1
        ]
        
        if len(nodes_to_cluster) < 2:
            return None
            
        positions = np.array([
            [self.G.nodes[n]['position']['x'], self.G.nodes[n]['position']['y']]
            for n in nodes_to_cluster
        ])
        
        clusters = await self.cluster_nodes(nodes_to_cluster, positions, level)
        if not clusters:
            return None
            
        # Create and summarize clusters
        for label, members in clusters.items():
            cluster_id = f"cluster_L{level}_{label}"
            await self._create_cluster(cluster_id, members, level)
        
        return clusters

    async def _create_cluster(self, cluster_id, members, level):
        """Create and summarize a single cluster"""
        member_nodes = [m[0] for m in members]
        member_positions = np.array([m[1] for m in members])
        center = np.mean(member_positions, axis=0)
        
        # Add cluster node
        self.G.add_node(
            cluster_id,
            type='cluster',
            level=level,
            members=member_nodes,
            position={
                'x': float(center[0]),
                'y': float(center[1]),
                'z': 0.0
            }
        )
        
        # Add edges
        for member in member_nodes:
            self.G.add_edge(cluster_id, member)
        
        # Generate summary
        cluster_text = self._prepare_cluster_text(member_nodes)
        summary_data = await self.llm.generate_cluster_summary(cluster_text)
        self.G.nodes[cluster_id].update(summary_data)

    def _prepare_cluster_text(self, members):
        """Prepare text description of cluster members"""
        texts = []
        for member in members:
            node_data = self.G.nodes[member]
            if node_data.get('type') == 'cluster':
                texts.append(f"Area: {node_data.get('name', 'Unnamed')}")
                texts.append(f"Description: {node_data.get('summary', 'No summary')}")
                rels = node_data.get('relationships', [])
                if rels:
                    texts.append("Related features: " + "; ".join(rels))
            else:
                texts.append(f"Location: {node_data.get('caption', 'No description')}")
        return "\n".join(texts)

