from enum import Enum

class Config:


    # LLM Parameters
    LLM = {
        'model': 'gpt-4o',
        'temperature': 0.7,
        'max_tokens': 500

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
            'k-branch': 3,
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

    # Location Parameters
    LOCATION = {
        'default_center': {
            'latitude': 40.4433,  # Pittsburgh
            'longitude': -79.9436
        },
        'search_radius': 500  # Default search radius in meters
    }



