from enum import Enum

class Config:

    # Path configurations
    PATHS = {
        'semantic_graphs_dir': 'semantic_graphs',  # Directory containing semantic graphs
        'latest_graph': 'enhanced_semantic_graph_semantic_graph_Building99_20241103_193232.gml',  # Your latest graph file
        'experiment_logs_dir': 'experiment_logs',  # Directory for experiment logs
        'graph_path': "/Users/danielxie/E-RAG/Embodied-RAG/graph",
        'vector_db_subdir': "vector_db",
        'semantic_forest_path': "/Users/danielxie/E-RAG/Embodied-RAG/graph/semantic_forests/graph/semantic_forest_graph.gml"
    }
    



    # LLM Parameters
    LLM = {
        'model': 'gpt-4o',
        'temperature': 0.7,
        'max_tokens': 500
        # 'vllm_settings': {
        #     'enabled': True,
        #     'model': 'meta-llama/Llama-3.2-3B-Instruct',
        #     'api_key': 'test-vllm',
        #     'api_base': 'http://localhost:8000',
        #     'swap_space': 4, 
        #     'max_num_seqs': 64,
        #     'tensor_parallel_size': 1,
        #     'server_startup_timeout': 120
        # }
    }

    # Graph Parameters
    GRAPH = {
        'drone_node_distance': 3.0,  # minimum distance between drone nodes
        'edge_types': {
            'spatial': ['north', 'south', 'east', 'west', 'above', 'below'],
            'hierarchical': ['part_of'],
        }
    }


    # Query Configuration
    QUERIES = {
        'implicit': {
            'default': "Where can I eat my lunch?",
            'examples': [
                "I need a place to work",
                "Where can I relax?",
                "I'm looking for a place to have a meeting"
            ]
        },
        'explicit': {
            'default': "Find the dining table",
            'examples': [
                "Navigate to the nearest chair",
                "Find the coffee table",
                "Locate the bookshelf"
            ]
        },
        'global': {
            'default': "What are the main types of furniture in this environment?",
            'examples': [
                "Describe the layout of this space",
                "What are the different functional areas?",
                "Give me an overview of this environment"
            ]
        }
    }

    # Spatial Analysis Parameters
    SPATIAL = {
        'cluster_distance_threshold': 20.0,  # Distance in meters
        'level_multiplier': 2.0,  # Each level's threshold is 2x the previous
        'min_cluster_size': 2,  # Minimum number of objects to form a cluster
        'max_cluster_size': 10,  # Maximum number of objects in a cluster
        'position_tolerance': 0.5  # Tolerance for position matching (in meters)
    }

    # Vector Database Parameters
    VECTOR_DB = {
        'collection_name': 'nodes_CMU_500_new',
        'embedding_dim': 1536,  # OpenAI embedding dimension
        'distance_metric': 'COSINE',
        'batch_size': 100,  # Number of nodes to batch index
        'batch_delay': 0.1,  # Delay between batches in seconds
    }
    
    # Retrieval Parameters
    RETRIEVAL = {
        'max_hierarchical_level': 5,
        'context_window_size': 8192,
        'search_params': {
            'limit': 50,  # Number of base nodes to retrieve
            'score_threshold': 0.3,  # Minimum similarity score
        },
        'reranking': {
            'semantic_weight': 0.6,
            'spatial_weight': 0.4,
            'top_k': 10  # Number of nodes to keep after reranking
        },
        'parallel_retrieval': {
            'k_branches': 3,  # Number of branches to explore for global queries
            'temperature': 0.7  # Temperature for node selection
        }
    }
    
    # Chat Parameters
    CHAT = {
        'history_length': 3,  # Number of previous interactions to keep
        'max_chars': 16000,  # Maximum characters in context
    }



