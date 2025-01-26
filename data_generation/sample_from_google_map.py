import googlemaps
import numpy as np
import folium
from datetime import datetime
import os
import requests
from math import radians, sin, cos, sqrt, atan2, degrees
import multiprocessing
from functools import partial
from googlemaps import Client
from googlemaps.roads import snap_to_roads
import networkx as nx
from collections import deque
import json
import aiohttp
import asyncio
from tqdm import tqdm
from PIL import Image
import io

GOOGLE_API_KEY = os.getenv('GOOGLE_MAPS_API_KEY')
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_MAPS_API_KEY environment variable is not set")

def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculate the distance between two points on Earth."""
    R = 6371000  # Earth's radius in meters
    
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1-a))
    return R * c


def save_checkpoint(points_data, processed_images, checkpoint_file='checkpoint.json', radius=0):
    """Save current progress to a checkpoint file."""
    checkpoint = {
        'points_data': {
            'points': points_data[0],
            'path_segments': points_data[1]
        },
        'processed_images': processed_images,
        'timestamp': datetime.now().isoformat()
    }
    
    with open(checkpoint_file, 'w') as f:
        json.dump(checkpoint, f, indent=4)
    print(f"Checkpoint saved to {checkpoint_file}")

def load_checkpoint(radius=0, metadata_dir=None):
    """Load progress from a checkpoint file."""
    if metadata_dir:
        checkpoint_file = f"{metadata_dir}/{radius}_checkpoint.json"
    else:
        checkpoint_file = f"{radius}_checkpoint.json"
        
    if os.path.exists(checkpoint_file):
        try:
            with open(checkpoint_file, 'r') as f:
                checkpoint = json.load(f)
            
            # Ensure points data is valid
            points = checkpoint['points_data']['points']
            path_segments = checkpoint['points_data']['path_segments']
            
            if not points:  # If points is empty or None
                print(f"Invalid checkpoint data in {checkpoint_file}")
                return None
                
            points_data = (points, path_segments)
            processed_images = checkpoint['processed_images']
            print(f"Loaded checkpoint from {checkpoint_file}")
            print(f"Resuming from {len(processed_images)} processed images")
            return points_data, processed_images
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Error loading checkpoint: {e}")
            return None
    return None

async def get_street_view_images_async(points_data, dataset_dirs):
    """Download and merge three regular Street View images"""
    points, _ = points_data
    processed_images = []
    
    async with aiohttp.ClientSession() as session:
        tasks = []
        for i, (lat, lon) in enumerate(tqdm(points, desc="Downloading images")):
            task = asyncio.create_task(
                download_street_view_async(session, lat, lon, i, dataset_dirs)
            )
            tasks.append(task)
            
            # Process in batches to respect rate limits
            if len(tasks) >= 20 or i == len(points) - 1:
                batch_results = await asyncio.gather(*tasks)
                processed_images.extend([r for r in batch_results if r is not None])
                tasks = []
                await asyncio.sleep(0.1)  # Rate limiting
    
    return processed_images

async def download_street_view_async(session, lat, lon, point_id, dataset_dirs):
    """Download four views (front, right, back, left) and save both merged and separate images"""
    try:
        metadata_params = {
            'location': f'{lat},{lon}',
            'key': GOOGLE_API_KEY
        }
        
        async with session.get("https://maps.googleapis.com/maps/api/streetview/metadata",
                             params=metadata_params) as response:
            metadata = await response.json()
            
            if metadata['status'] != 'OK':
                return None
            
            # Use dataset directories
            merged_dir = dataset_dirs['images']
            separate_dir = dataset_dirs['separate']
            
            pano_id = metadata.get('pano_id')
            heading = metadata.get('heading', 0)
            
            # Download four views (front, right, back, left)
            images_to_merge = []
            separate_images = []
            headings = []
            
            # Using 0° (front), 90° (right), 180° (back), and 270° (left) angles
            for angle in [0, 90, 180, 270]:
                params = {
                    'location': f'{lat},{lon}',
                    'size': '640x640',
                    'heading': (heading + angle) % 360,
                    'pitch': 0,
                    'key': GOOGLE_API_KEY
                }
                
                async with session.get("https://maps.googleapis.com/maps/api/streetview",
                                     params=params) as img_response:
                    if img_response.status == 200:
                        # Read image data
                        img_data = await img_response.read()
                        img = Image.open(io.BytesIO(img_data))
                        images_to_merge.append(img)
                        
                        # Save separate image
                        separate_filename = f'{separate_dir}/streetview_{point_id:04d}_{angle}.jpg'
                        img.save(separate_filename, 'JPEG', quality=95)
                        
                        separate_images.append({
                            'path': separate_filename,
                            'heading': (heading + angle) % 360,
                            'angle': angle
                        })
                        headings.append((heading + angle) % 360)
            
            if len(images_to_merge) == 4:
                # Create merged image (4 images side by side)
                merged_width = 640 * 4
                merged_height = 640
                merged_image = Image.new('RGB', (merged_width, merged_height))
                
                # Paste images side by side (left, front, right, back)
                image_order = [3, 0, 1, 2]  # 270° (left), 0° (front), 90° (right), 180° (back)
                for i, img_idx in enumerate(image_order):
                    merged_image.paste(images_to_merge[img_idx], (i * 640, 0))
                
                # Save merged image
                merged_filename = f'{merged_dir}/streetview_{point_id:04d}.jpg'
                merged_image.save(merged_filename, 'JPEG', quality=95)
                
                return {
                    'point_id': point_id,
                    'latitude': lat,
                    'longitude': lon,
                    'pano_id': pano_id,
                    'heading': heading,
                    'date': metadata.get('date', ''),
                    'merged_image': {
                        'path': merged_filename,
                        'headings': [headings[3], headings[0], headings[1], headings[2]]  # left, front, right, back
                    },
                    'separate_images': separate_images,
                    'status': 'OK'
                }
            
    except Exception as e:
        print(f"Error downloading images for point {point_id}: {e}")
    return None

def visualize_points(center_lat, center_lon, points_data, save_path=None):
    """Visualize sampling points and their connections on a map"""
    points, _ = points_data
    
    # Create map centered on the first point
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=19,
        tiles='https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}',
        attr='Google Satellite'
    )
    
    # Add points to map
    for i, (lat, lon) in enumerate(points):
        folium.CircleMarker(
            location=[lat, lon],
            radius=3,
            color='blue',
            fill=True,
            popup=f'Point {i}'
        ).add_to(m)
    
    # Add connections between nearby points
    max_connection_distance = 15  # meters, same as in save_combined_data
    for i, (lat1, lon1) in enumerate(points):
        for j, (lat2, lon2) in enumerate(points):
            if i < j:  # Only process each pair once
                distance = haversine_distance(lat1, lon1, lat2, lon2)
                if distance <= max_connection_distance:
                    # Draw a line between connected points
                    folium.PolyLine(
                        locations=[[lat1, lon1], [lat2, lon2]],
                        weight=2,
                        color='red',
                        opacity=0.5
                    ).add_to(m)
    
    if save_path:
        m.save(save_path)

def save_combined_data(points_data, images_data, save_path='streetview_graph.json', batch_size=20):
    """Save combined graph data without connectivity restrictions"""
    print("Processing graph data...")
    
    # Initialize the combined data structure
    combined_data = {
        "metadata": {
            "total_points": len(images_data),
            "timestamp": datetime.now().isoformat()
        },
        "nodes": []
    }
    
    # Add nodes without connectivity information
    for image_data in images_data:
        node_data = {
            "id": image_data['point_id'],
            "location": {
                "latitude": image_data['latitude'],
                "longitude": image_data['longitude']
            },
            "images": {
                "merged": image_data['merged_image'],
                "separate": image_data['separate_images']
            },
            "date": image_data.get('date', ''),
            "pano_id": image_data['pano_id'],
            "base_heading": image_data['heading'],
            "status": image_data['status']
        }
        combined_data["nodes"].append(node_data)
    
    # Save to JSON file
    with open(save_path, 'w') as f:
        json.dump(combined_data, f, indent=4)
    
    print(f"Combined data saved to {save_path}")
    return combined_data


async def get_street_view_coverage(center_lat, center_lon, radius, spacing=15):
    """Sample actual Street View panorama locations within radius"""
    print("Finding Street View panoramas...")
    
    # Calculate bounds
    lat_offset = radius / 111320
    lon_offset = radius / (111320 * cos(radians(center_lat)))
    
    lat_min = center_lat - lat_offset
    lat_max = center_lat + lat_offset
    lon_min = center_lon - lon_offset
    lon_max = center_lon + lon_offset
    
    # Generate initial grid points
    search_spacing = spacing / 2
    lat_step = search_spacing / 111320
    lon_step = search_spacing / (111320 * cos(radians(center_lat)))
    
    # Calculate grid points
    search_points = []
    lat = lat_min
    while lat <= lat_max:
        lon = lon_min
        while lon <= lon_max:
            if haversine_distance(center_lat, center_lon, lat, lon) <= radius:
                search_points.append((lat, lon))
            lon += lon_step
        lat += lat_step
    
    # Estimation factors
    street_coverage_factor = 0.6  # About 60% of points have Street View data
    avg_image_size_kb = 100      # Average size per image
    views_per_point = 4          # Front, right, back, left views
    
    # Updated pricing tiers
    tier1_cost = 0.007  # $0.007 per request (0-100K)
    tier2_cost = 0.0056 # $0.0056 per request (100K-500K)
    
    # Calculate estimates
    estimated_points = int(len(search_points) * street_coverage_factor)
    total_image_requests = estimated_points * views_per_point
    estimated_storage_mb = (estimated_points * views_per_point * avg_image_size_kb) / 1024
    
    # Calculate API costs with tiered pricing
    if total_image_requests <= 500000:  # Up to 500K
        api_cost = total_image_requests * tier2_cost
    else:
        tier2_requests = 500000
        tier1_requests = total_image_requests - 500000
        api_cost = (tier2_requests * tier2_cost) + (tier1_requests * tier1_cost)
    
    # Print estimates
    print("\nEstimation Summary:")
    print(f"Grid points: {len(search_points):,}")
    print(f"Estimated Street View points: {estimated_points:,}")
    print(f"Total image requests: {total_image_requests:,}")
    print("\nStorage Requirements:")
    print(f"Estimated storage needed: {estimated_storage_mb:.2f} MB")
    print(f"Estimated storage needed: {(estimated_storage_mb/1024):.2f} GB")
    print("\nAPI Costs:")
    if total_image_requests > 500000:
        print(f"  - Tier 2 ($0.0056/req): 500,000")
        print(f"  - Tier 1 ($0.007/req): {total_image_requests - 500000:,}")
    else:
        print(f"  - Tier 2 ($0.0056/req): {total_image_requests:,}")
    print(f"Estimated API cost: ${api_cost:,.2f}")
    print("\nNote: Each point requires 4 Street View images (front, right, back, left)")
    print("      Actual numbers may vary based on street coverage density")
    
    user_input = input("\nWould you like to continue? (yes/no): ")
    if user_input.lower() != 'yes':
        return None, None
    
    # Find actual panorama locations
    valid_points = []
    seen_pano_ids = set()
    
    async with aiohttp.ClientSession() as session:
        batch_size = 50
        for i in tqdm(range(0, len(search_points), batch_size), desc="Finding panoramas"):
            batch = search_points[i:i + batch_size]
            tasks = []
            
            for lat, lon in batch:
                params = {
                    'location': f'{lat},{lon}',
                    'key': GOOGLE_API_KEY
                }
                tasks.append(get_panorama_metadata(session, lat, lon, params))
            
            # Add delay between batches to respect rate limits
            if i > 0:
                await asyncio.sleep(0.1)
            
            try:
                results = await asyncio.gather(*tasks)
                
                # Process results with logging
                for result in results:
                    if result and result.get('pano_id'):
                        pano_id = result['pano_id']
                        if pano_id not in seen_pano_ids:
                            seen_pano_ids.add(pano_id)
                            lat = result['location']['lat']
                            lng = result['location']['lng']
                            valid_points.append((lat, lng))
                            
                # Save intermediate progress every 1000 points
                if len(valid_points) > 0 and len(valid_points) % 1000 == 0:
                    print(f"\nFound {len(valid_points)} valid points so far...")
                    save_checkpoint((valid_points, []), [], f"intermediate_checkpoint_{len(valid_points)}.json")
                    
            except Exception as e:
                print(f"Error processing batch {i}: {e}")
                continue
                
        print(f"\nFound {len(valid_points)} unique panorama locations")
        if len(valid_points) == 0:
            print("Warning: No valid panorama locations found. This might indicate an API issue.")
            return None, None
            
        return valid_points, []

async def get_panorama_metadata(session, lat, lon, params):
    """Get panorama metadata for a location with error handling and logging"""
    try:
        async with session.get("https://maps.googleapis.com/maps/api/streetview/metadata",
                             params=params) as response:
            metadata = await response.json()
            
            # Add debug logging
            # print(f"Metadata response for {lat},{lon}: {metadata}")
            
            if metadata.get('status') == 'OK':
                return {
                    'pano_id': metadata.get('pano_id'),
                    'location': {
                        'lat': metadata.get('location', {}).get('lat'),
                        'lng': metadata.get('location', {}).get('lng')
                    }
                }
            else:
                print(f"Invalid status for {lat},{lon}: {metadata.get('status')}")
                return None
                
    except Exception as e:
        print(f"Error getting metadata for {lat},{lon}: {e}")
        return None

def create_dataset_directories(dataset_name):
    """Create directory structure for the dataset"""
    base_dir = f'datasets/{dataset_name}'
    dirs = {
        'images': f'{base_dir}/images/merged',
        'separate': f'{base_dir}/images/separate',
        'metadata': f'{base_dir}/metadata',
        'visualization': f'{base_dir}/visualization'
    }
    for dir_path in dirs.values():
        os.makedirs(dir_path, exist_ok=True)
    return dirs

def main():
    # Get dataset name from user
    dataset_name = input("Enter a name for this dataset: ").strip()
    if not dataset_name:
        print("Dataset name cannot be empty")
        return
    
    # Create dataset directories
    dataset_dirs = create_dataset_directories(dataset_name)
    
    # # # Tokyo Station
    # center_lat = 35.6812
    # center_lon = 139.7671
    # radius = 1000  # meters
    # spacing = 5   # meters between points

    # Pittsburgh
    center_lat = 40.4433
    center_lon = -79.9436
    radius = 500  # meters
    spacing = 10   # meters between points
    
    # Update file paths for dataset
    checkpoint_file = f'{dataset_dirs["metadata"]}/{radius}_checkpoint.json'
    graph_file = f'{dataset_dirs["metadata"]}/streetview_graph.json'
    map_file = f'{dataset_dirs["visualization"]}/sampling_points_map.html'
    
    # Check for existing checkpoint
    checkpoint_data = load_checkpoint(radius, dataset_dirs["metadata"])
    if checkpoint_data and checkpoint_data[0] and checkpoint_data[0][0]:
        points_data, processed_images = checkpoint_data
        print("Resuming from checkpoint...")
    else:
        print("No valid checkpoint found. Starting new collection...")
        points_data = asyncio.run(get_street_view_coverage(
            center_lat, 
            center_lon, 
            radius,
            spacing=spacing
        ))
        if points_data is None:
            print("Operation cancelled.")
            return
        processed_images = []
        
        # Save initial checkpoint
        save_checkpoint(points_data, processed_images, checkpoint_file)
    
    if points_data and points_data[0]:
        # Visualize points with path information
        visualize_points(center_lat, center_lon, points_data, save_path=map_file)
        
        print(f"\nFound {len(points_data[0])} total sampling points")
        print(f"Already processed: {len(processed_images)} images")
        
        user_input = input("Would you like to proceed with downloading images? (yes/no): ")
        if user_input.lower() == 'yes':
            # Pass dataset directories to download function
            images_data = asyncio.run(get_street_view_images_async(points_data, dataset_dirs))
            
            # Save combined data
            combined_data = save_combined_data(
                points_data, 
                images_data, 
                save_path=graph_file,
                batch_size=20
            )
            
            print(f"Dataset '{dataset_name}' created successfully!")
            print(f"- Images: datasets/{dataset_name}/images/")
            print(f"- Metadata: datasets/{dataset_name}/metadata/")
            print(f"- Visualization: datasets/{dataset_name}/visualization/")
            
            if os.path.exists(checkpoint_file):
                os.remove(checkpoint_file)
        else:
            print("Image download cancelled.")
    else:
        print("No valid points data found. Please try again.")

if __name__ == "__main__":
    main()


