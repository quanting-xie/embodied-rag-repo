import numpy as np
from math import cos, pi, radians, sin, atan2, sqrt
import json
import random
import asyncio
from openai import AsyncOpenAI
import os
from typing import List, Dict

class QueryEnhancer:
    def __init__(self):
        self.client = AsyncOpenAI(
            api_key=os.getenv('OPENAI_API_KEY')
        )
    
    async def generate_searchable_term(self, query: str) -> str:
        """Generate the most relevant search term for Google Maps from a query."""
        prompt = f"""Given this location query: "{query}"
        Generate ONE specific search term that would be most suitable for searching on Google Maps.
        The term should:
        1. Directly match the user's search intent
        2. Use terminology commonly found on Google Maps
        3. Be specific enough to find relevant results
        
        Examples:
        Query: "Find me a coffee shop" -> "coffee shop"
        Query: "Where can I buy groceries?" -> "supermarket"
        Query: "I need to withdraw some cash" -> "ATM"
        Query: "Looking for a place to work" -> "coworking space"
        
        Provide only the search term without quotes or explanation.
        Your response for "{query}":"""
        
        try:
            response = await self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that generates precise search terms for Google Maps."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3  # Lower temperature for more focused results
            )
            
            # Get the single search term
            search_term = response.choices[0].message.content.strip()
            return search_term
            
        except Exception as e:
            print(f"Error generating search term for '{query}': {str(e)}")
            return query  # Fallback to original query

async def enhance_queries(query_locations: List[Dict]) -> List[Dict]:
    """Add searchable term to each query location."""
    enhancer = QueryEnhancer()
    enhanced_locations = []
    
    print("\nGenerating searchable terms for queries...")
    for i, query_item in enumerate(query_locations):
        search_term = await enhancer.generate_searchable_term(query_item["query"])
        enhanced_locations.append({
            **query_item,
            "searchable_term": search_term  # Changed from plural to singular
        })
        print(f"Processed {i+1}/{len(query_locations)} queries: '{query_item['query']}' -> '{search_term}'")
        
    return enhanced_locations

def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculate the distance between two points on Earth."""
    R = 6371000  # Earth's radius in meters
    
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1-a))
    return R * c

def generate_random_coordinates(center_lat, center_lon, radius_meters, num_points):
    """Generate random coordinates within a radius."""
    coordinates = []
    
    # Convert radius from meters to degrees (approximate)
    radius_degrees = radius_meters / 111300  # 1 degree is approximately 111.3 km
    
    # Generate points until we have enough valid ones
    while len(coordinates) < num_points:
        # Generate random angle and radius
        theta = np.random.uniform(0, 2 * pi)
        r = np.random.uniform(0, radius_degrees)
        
        # Convert to lat/lon offset
        dlat = r * np.cos(theta)
        dlon = r * np.sin(theta) / np.cos(radians(center_lat))
        
        # Calculate new point
        new_lat = center_lat + dlat
        new_lon = center_lon + dlon
        
        # Verify distance is within radius
        distance = haversine_distance(center_lat, center_lon, new_lat, new_lon)
        if distance <= radius_meters:
            coordinates.append([float(new_lat), float(new_lon)])
    
    return coordinates

def load_queries(file_path):
    """Load queries from text file."""
    with open(file_path, 'r') as f:
        return [line.strip() for line in f if line.strip()]

def assign_coordinates_to_queries(queries, coordinates):
    """Assign random coordinates to queries without replacement."""
    # Ensure we have enough coordinates
    if len(coordinates) < len(queries):
        raise ValueError(f"Not enough coordinates ({len(coordinates)}) for queries ({len(queries)})")
    
    # Randomly assign coordinates to queries
    assigned_coordinates = random.sample(coordinates, len(queries))
    
    # Create query-coordinate pairs
    query_locations = []
    for query, coord in zip(queries, assigned_coordinates):
        query_locations.append({
            "query": query,
            "location": {
                "latitude": coord[0],
                "longitude": coord[1]
            }
        })
    
    return query_locations

async def main():
    # Input parameters
    center_lat = 35.6812  # Tokyo Station
    center_lon = 139.7671
    radius_meters = 1000
    queries_file = 'location_queries.txt'
    output_file = 'query_locations.json'
    
    # Load queries
    print("Loading queries...")
    queries = load_queries(queries_file)
    num_queries = len(queries)
    
    # Generate coordinates
    print(f"Generating {num_queries} coordinates within {radius_meters}m of {center_lat}, {center_lon}")
    coordinates = generate_random_coordinates(center_lat, center_lon, radius_meters, num_queries)
    
    # Assign coordinates to queries
    print("Assigning coordinates to queries...")
    query_locations = assign_coordinates_to_queries(queries, coordinates)
    
    # Enhance queries with searchable terms
    enhanced_locations = await enhance_queries(query_locations)
    
    # Save results
    with open(output_file, 'w') as f:
        json.dump({
            "metadata": {
                "center": {
                    "latitude": center_lat,
                    "longitude": center_lon
                },
                "radius_meters": radius_meters,
                "total_queries": len(enhanced_locations)
            },
            "query_locations": enhanced_locations
        }, f, indent=2)
    
    print(f"\nEnhanced query locations saved to {output_file}")
    print(f"Total assignments: {len(enhanced_locations)}")

if __name__ == "__main__":
    asyncio.run(main()) 