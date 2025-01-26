import numpy as np
import networkx as nx
from sklearn.cluster import AgglomerativeClustering
import re
from config import Config
import traceback
from math import radians, sin, cos, sqrt, atan2
import asyncio

class SemanticClusterer:
    def __init__(self, llm_interface):
        self.llm = llm_interface
        self.base_threshold = 10  # Reduced base distance threshold
        self.level_multiplier = 1.2
        self.max_levels = Config.RETRIEVAL['max_hierarchical_level']
        self.semantic_weight = 0.6  # Increased semantic weight
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
        
        print(f"\nProcessing {n_nodes} nodes for similarity matrix...")
        
        # Process in smaller chunks to show progress
        chunk_size = 1000
        for i in range(0, n_nodes, chunk_size):
            chunk_end = min(i + chunk_size, n_nodes)
            print(f"\nProcessing nodes {i} to {chunk_end} of {n_nodes}")
            
            # Get node texts for this chunk
            node_texts = []
            for node_id in nodes[i:chunk_end]:
                node_data = self.G.nodes[node_id]
                
                if node_data.get('type') == 'cluster':
                    summary = node_data.get('summary', '').strip()
                    name = node_data.get('name', '').strip()
                    text = summary if summary else name
                else:
                    name = node_data.get('name', '').strip()
                    caption = node_data.get('caption', '').strip()
                    text = f"{name}. {caption}" if name else caption
                
                text = text.strip()
                if not text:
                    text = "unnamed location"
                node_texts.append(text)
            
            try:
                # Get embeddings for this chunk
                chunk_embeddings = await self.llm.generate_embeddings(node_texts)
                
                # Compute similarities for this chunk
                for j in range(len(chunk_embeddings)):
                    for k in range(n_nodes):
                        if k >= i + j:  # Only compute upper triangle
                            # Calculate spatial similarity
                            lat1, lon1 = positions[i + j][1], positions[i + j][0]
                            lat2, lon2 = positions[k][1], positions[k][0]
                            spatial_dist = self.haversine_distance(lat1, lon1, lat2, lon2)
                            spatial_sim = np.exp(-spatial_dist / self.base_threshold)
                            
                            # Calculate semantic similarity if we have both embeddings
                            if k < chunk_end:
                                semantic_sim = np.dot(chunk_embeddings[j], chunk_embeddings[k-i]) / (
                                    np.linalg.norm(chunk_embeddings[j]) * np.linalg.norm(chunk_embeddings[k-i]))
                            else:
                                semantic_sim = 0.5  # Default similarity for nodes we haven't processed yet
                            
                            # Combine similarities
                            combined_sim = (1 - self.semantic_weight) * spatial_sim + self.semantic_weight * semantic_sim
                            
                            similarity_matrix[i + j, k] = combined_sim
                            similarity_matrix[k, i + j] = combined_sim
                
                print(f"Completed chunk {i//chunk_size + 1}/{(n_nodes + chunk_size - 1)//chunk_size}")
                
            except Exception as e:
                print(f"Error processing chunk {i//chunk_size + 1}: {str(e)}")
                # Use spatial similarity only for failed chunks
                for j in range(len(node_texts)):
                    for k in range(n_nodes):
                        if k >= i + j:
                            lat1, lon1 = positions[i + j][1], positions[i + j][0]
                            lat2, lon2 = positions[k][1], positions[k][0]
                            spatial_dist = self.haversine_distance(lat1, lon1, lat2, lon2)
                            spatial_sim = np.exp(-spatial_dist / self.base_threshold)
                            
                            similarity_matrix[i + j, k] = spatial_sim
                            similarity_matrix[k, i + j] = spatial_sim
        
        # Set diagonal to 1
        np.fill_diagonal(similarity_matrix, 1.0)
        
        return similarity_matrix

    async def cluster_nodes(self, nodes, positions, level):
        """Perform hierarchical clustering on nodes"""
        if len(nodes) < 2:
            return []
        
        print(f"\nComputing similarity matrix for {len(nodes)} nodes...")
        # Get combined similarity matrix
        similarity_matrix = await self._compute_similarity_matrix(nodes, positions)
        print("Similarity matrix computed.")
        
        # Convert similarity to distance
        distance_matrix = 1 - similarity_matrix
        
        # Calculate clustering threshold
        distance_threshold = 0.4
        if level > 1:
            distance_threshold *= (1 + (level - 1) * 0.1)
        
        print(f"\nPerforming hierarchical clustering with threshold {distance_threshold:.3f}...")
        # Perform clustering
        clustering = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=distance_threshold,
            metric='precomputed',
            linkage='complete'
        )
        
        try:
            labels = clustering.fit_predict(distance_matrix)
            
            # Group nodes by cluster
            clusters = {}
            for i, label in enumerate(labels):
                if label not in clusters:
                    clusters[label] = []
                clusters[label].append((nodes[i], positions[i]))
            
            print(f"\nClustering Results:")
            print(f"- Created {len(clusters)} clusters")
            sizes = [len(c) for c in clusters.values()]
            if sizes:
                print(f"- Cluster sizes: min={min(sizes)}, max={max(sizes)}, mean={np.mean(sizes):.1f}")
                print(f"- Size distribution: {np.percentile(sizes, [25, 50, 75])}")
            
            return clusters
            
        except Exception as e:
            print(f"Clustering failed: {e}")
            return []

    async def extract_relationships(self, objects):
        """Extract relationships between objects and build hierarchical graph"""
        # Initialize the graph
        self.G = nx.Graph()
        
        # Add base nodes first
        print("Adding base nodes...")
        for obj in objects:
            self.G.add_node(obj['id'], 
                          type='base',
                          position=obj['position'],
                          name=obj['name'],
                          caption=obj['caption'],
                          timestamp=obj['timestamp'],
                          image_path=obj['image_path'],  # Make sure this is included
                          level=0)
        
        # Iterative clustering
        current_level = 1
        while current_level <= self.max_levels:
            print(f"\nLevel {current_level} clustering:")
            print("----------------------------------------")
            clusters = await self._process_level(current_level)
            if not clusters:
                break
            current_level += 1
        
        # Debug print graph info before returning
        print(f"\nFinal graph summary:")
        print(f"- Total nodes: {len(self.G.nodes())}")
        print(f"- Base nodes: {len([n for n, d in self.G.nodes(data=True) if d.get('type') == 'base'])}")
        print(f"- Cluster nodes: {len([n for n, d in self.G.nodes(data=True) if d.get('type') == 'cluster'])}")
        
        # Return the graph
        return self.G

    async def _process_level(self, level):
        """Process a single level of clustering"""
        print(f"\nProcessing Level {level}")
        print("----------------------------------------")
        
        # 1. Get nodes to cluster
        nodes_to_cluster = [
            node for node, data in self.G.nodes(data=True)
            if data.get('level', 0) == level - 1
        ]
        print(f"Found {len(nodes_to_cluster)} nodes to cluster")
        
        if len(nodes_to_cluster) < 2:
            return None
            
        # 2. Get positions
        print("Preparing position data...")
        positions = np.array([
            [self.G.nodes[n]['position']['x'], self.G.nodes[n]['position']['y']]
            for n in nodes_to_cluster
        ])
        
        # 3. Perform clustering
        print(f"Clustering {len(nodes_to_cluster)} nodes...")
        clusters = await self.cluster_nodes(nodes_to_cluster, positions, level)
        if not clusters:
            return None
        
        # 4. Process clusters in smaller batches
        batch_size = 10  # Reduced batch size
        cluster_items = list(clusters.items())
        total_clusters = len(cluster_items)
        
        print(f"\nProcessing {total_clusters} clusters in batches of {batch_size}")
        processed_clusters = 0
        
        try:
            for i in range(0, total_clusters, batch_size):
                batch = cluster_items[i:i + batch_size]
                batch_end = min(i + batch_size, total_clusters)
                print(f"\nProcessing batch {i//batch_size + 1}/{(total_clusters + batch_size - 1)//batch_size}")
                print(f"Clusters {i} to {batch_end} of {total_clusters}")
                
                # Process batch with timeout
                try:
                    tasks = []
                    for label, members in batch:
                        cluster_id = f"cluster_L{level}_{label}"
                        task = self._create_cluster(cluster_id, members, level)
                        tasks.append(task)
                    
                    # Wait for batch with timeout
                    await asyncio.wait_for(asyncio.gather(*tasks), timeout=300)  # 5 minute timeout
                    processed_clusters += len(batch)
                    print(f"✓ Processed {processed_clusters}/{total_clusters} clusters")
                    
                    # Add small delay between batches
                    await asyncio.sleep(0.5)
                    
                except asyncio.TimeoutError:
                    print(f"⚠️ Batch {i//batch_size + 1} timed out, continuing with next batch")
                    continue
                except Exception as e:
                    print(f"⚠️ Error in batch {i//batch_size + 1}: {str(e)}")
                    print(traceback.format_exc())
                    continue
        
        except Exception as e:
            print(f"Error during cluster processing: {str(e)}")
            print(traceback.format_exc())
        
        finally:
            print(f"\nLevel {level} processing complete:")
            print(f"- Total clusters: {total_clusters}")
            print(f"- Successfully processed: {processed_clusters}")
            print(f"- Failed: {total_clusters - processed_clusters}")
        
        return clusters

    async def _create_cluster(self, cluster_id, members, level):
        """Create and summarize a single cluster"""
        try:
            member_nodes = [m[0] for m in members]
            member_positions = np.array([m[1] for m in members])
            center = np.mean(member_positions, axis=0)
            
            # Add cluster node with basic info
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
            
            # Only update if summary generation was successful
            if summary_data:
                self.G.nodes[cluster_id].update(summary_data)
                print(f"✓ Created cluster {cluster_id}: {summary_data['name']}")
            else:
                print(f"⚠️ Failed to generate summary for cluster {cluster_id}")
            
        except Exception as e:
            print(f"Error creating cluster {cluster_id}: {str(e)}")
            raise

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

