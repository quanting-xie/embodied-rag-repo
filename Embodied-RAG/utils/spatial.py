import numpy as np
from typing import Dict, Optional
import re
import traceback

def extract_location_from_query(query: str) -> Optional[Dict[str, float]]:
    """Extract location coordinates from query text using regex"""
    parts = [part.strip() for part in query.split('|')]
    
    for part in parts:
        if part.startswith('L:'):
            try:
                coords = part[2:].strip()
                lat_str, lon_str = coords.split(',')
                lat, lon = float(lat_str.strip()), float(lon_str.strip())
                
                if -90 <= lat <= 90 and -180 <= lon <= 180:
                    return {'latitude': lat, 'longitude': lon}
            except (ValueError, IndexError):
                continue
    
    return None

def normalize_location_format(location: Dict) -> Dict[str, float]:
    """Normalize location format to use latitude/longitude keys"""
    return {
        'latitude': float(location.get('y', location.get('latitude', 0))),
        'longitude': float(location.get('x', location.get('longitude', 0)))
    }

def compute_haversine_distance(loc1: Dict, loc2: Dict) -> float:
    """
    Compute haversine distance between two locations in meters
    
    Args:
        loc1, loc2: Dictionaries containing latitude/longitude or x/y coordinates
    """
    # Normalize coordinate format
    loc1 = normalize_location_format(loc1)
    loc2 = normalize_location_format(loc2)
    
    R = 6371000  # Earth's radius in meters
    
    phi1, phi2 = np.radians(loc1['latitude']), np.radians(loc2['latitude'])
    dphi = np.radians(loc2['latitude'] - loc1['latitude'])
    dlambda = np.radians(loc2['longitude'] - loc1['longitude'])
    
    a = np.sin(dphi/2)**2 + \
        np.cos(phi1) * np.cos(phi2) * np.sin(dlambda/2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    
    return R * c

def compute_spatial_score(query_location: Dict, node_location: Dict, max_distance: float = 2000.0) -> float:
    """
    Compute spatial relevance score using linear normalization
    Args:
        query_location: Dict with latitude and longitude
        node_location: Dict with latitude and longitude
        max_distance: Maximum distance in meters (default 2000m)
    Returns:
        float: Normalized score between 0 and 1 (1 = same location, 0 = max_distance or further)
    """
    try:
        distance = compute_haversine_distance(query_location, node_location)
        # Linear normalization: 1 - (distance / max_distance), clamped to [0, 1]
        score = max(0.0, min(1.0, 1.0 - (distance / max_distance)))
        return score
    except Exception as e:
        return 0.0 