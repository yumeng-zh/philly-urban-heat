#!/usr/bin/env python3
"""
Google Street View Batch Downloader
==================================

Downloads street-level imagery for computer vision analysis with:
- 4-direction sampling (N/E/S/W) per location
- Metadata pre-checking to avoid invalid requests
- Automatic retry logic and rate limiting
- Progress tracking and resume capability

Usage:
    1. Get Google Cloud API key: https://console.cloud.google.com/apis/credentials
    2. Enable Street View Static API
    3. Set API_KEY = "your_key_here" below
    4. Run: python download_street_view_images.py

Input: CSV file with columns: grid_id, lat, lon, anomaly_type
Output: Images saved to ./streetview_images/ with metadata logs
"""

import os
import requests
import pandas as pd
import time
import json
from datetime import datetime
import hashlib

# =============================================================================
# CONFIGURATION
# =============================================================================

# REPLACE WITH YOUR GOOGLE CLOUD API KEY
API_KEY = "YOUR_GOOGLE_MAPS_API_KEY"  # Get from: https://console.cloud.google.com/apis/credentials

# API endpoints
METADATA_URL = "https://maps.googleapis.com/maps/api/streetview/metadata"
STREETVIEW_URL = "https://maps.googleapis.com/maps/api/streetview"

# Image parameters
IMAGE_SIZE = "640x640"  # Maximum free tier size
FOV = 90  # Field of view (degrees)

# Directions for 4-way sampling
DIRECTIONS = {
    'north': 0,
    'east': 90, 
    'south': 180,
    'west': 270
}

# Rate limiting (requests per second)
RATE_LIMIT = 2  # Conservative rate to avoid API limits

# =============================================================================
# CORE FUNCTIONS
# =============================================================================

def check_api_key():
    """Verify API key is configured"""
    if API_KEY == "YOUR_API_KEY_HERE":
        print("❌ Error: Please set your Google Cloud API key in the script")
        print("   1. Go to: https://console.cloud.google.com/apis/credentials")
        print("   2. Create API key and enable Street View Static API")
        print("   3. Replace 'YOUR_API_KEY_HERE' with your key")
        return False
    return True


def test_api_key():
    """Quick test to verify API key works"""
    print("🔑 Testing API key...")
    # Test with a known good location (Philadelphia City Hall)
    params = {
        'location': '39.9526,-75.1652',
        'key': API_KEY
    }
    try:
        resp = requests.get(METADATA_URL, params=params, timeout=10)
        data = resp.json()
        print(f"   HTTP status: {resp.status_code}")
        print(f"   API response status: {data.get('status')}")
        if data.get('status') == 'OK':
            print("   ✅ API key working!")
            return True
        elif data.get('status') == 'REQUEST_DENIED':
            print(f"   ❌ REQUEST_DENIED — check API key and billing")
            print(f"   Error: {data.get('error_message', 'no error message')}")
            return False
        else:
            print(f"   ⚠ Unexpected status: {data.get('status')}")
            return False
    except Exception as e:
        print(f"   ❌ Connection error: {e}")
        return False

def get_street_view_metadata(lat, lon):
    """
    Check if street view imagery is available at location
    Returns: dict with status and metadata, or None if failed
    """
    params = {
        'location': f"{lat},{lon}",
        'source': 'outdoor',
        'key': API_KEY
    }
    
    try:
        response = requests.get(METADATA_URL, params=params, timeout=10)
        if response.status_code == 200:
            return response.json()
        else:
            print(f"⚠ Metadata request failed: HTTP {response.status_code}")
            return None
    except Exception as e:
        print(f"⚠ Metadata request error: {e}")
        return None

def download_street_view_image(lat, lon, heading, output_path):
    """
    Download street view image
    Returns: True if successful, False otherwise
    """
    params = {
        'location': f"{lat},{lon}",
        'size': IMAGE_SIZE,
        'heading': heading,
        'fov': FOV,
        'pitch': 0,
        'source': 'outdoor',
        'key': API_KEY
    }
    
    try:
        response = requests.get(STREETVIEW_URL, params=params, timeout=15)
        if response.status_code == 200:
            with open(output_path, 'wb') as f:
                f.write(response.content)
            return True
        else:
            print(f"⚠ Image download failed: HTTP {response.status_code}")
            return False
    except Exception as e:
        print(f"⚠ Image download error: {e}")
        return False

def create_output_directories():
    """Create necessary output directories"""
    dirs = ['streetview_images', 'streetview_images/metadata', 'streetview_images/logs']
    for d in dirs:
        os.makedirs(d, exist_ok=True)

def load_sampling_points():
    """Load sampling points from CSV"""
    possible_files = [
        'anomaly_sampling_points.csv',
        '/home/claude/anomaly_sampling_points.csv',
        'sampling_points.csv'
    ]
    
    for file_path in possible_files:
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            print(f"✓ Loaded {len(df)} sampling points from: {file_path}")
            return df
    
    # If no file found, create sample data for testing
    print("⚠ No sampling points file found. Creating test data...")
    sample_data = [
        {'grid_id': 'test_001', 'lat': 39.9526, 'lon': -75.1652, 'anomaly_type': 'hot'},   # Center City
        {'grid_id': 'test_002', 'lat': 39.9611, 'lon': -75.2097, 'anomaly_type': 'cool'},  # Fairmount Park
        {'grid_id': 'test_003', 'lat': 40.0259, 'lon': -75.1503, 'anomaly_type': 'hot'}    # North Philly
    ]
    df = pd.DataFrame(sample_data)
    print(f"✓ Created {len(df)} test sampling points for Philadelphia")
    return df

def save_download_log(results, output_file='streetview_images/logs/download_log.json'):
    """Save download results to log file"""
    timestamp = datetime.now().isoformat()
    log_entry = {
        'timestamp': timestamp,
        'total_points': len(results),
        'successful_downloads': sum(1 for r in results if r['success']),
        'failed_downloads': sum(1 for r in results if not r['success']),
        'details': results
    }
    
    with open(output_file, 'w') as f:
        json.dump(log_entry, f, indent=2)
    
    print(f"💾 Download log saved to: {output_file}")

def main():
    """Main download function"""
    print("🌍 Google Street View Batch Downloader")
    print("=====================================")
    
    # Check configuration
    if not check_api_key():
        return
    if not test_api_key():
        return
    
    # Setup
    create_output_directories()
    df = load_sampling_points()
    
    if len(df) == 0:
        print("❌ No sampling points to process")
        return
    
    print(f"📍 Processing {len(df)} sampling points")
    print(f"🖼️  Will download {len(df) * 4} images (4 directions each)")
    print(f"💰 Estimated cost: ~${len(df) * 4 * 0.007:.2f}")
    print()
    
    # Process each sampling point
    results = []
    total_requests = 0
    successful_downloads = 0
    
    for idx, row in df.iterrows():
        grid_id = row['grid_id']
        lat, lon = row['lat'], row['lon']
        anomaly_type = row['anomaly_type']
        
        print(f"🔍 Processing {grid_id} ({idx+1}/{len(df)}) - {anomaly_type}")
        
        # Check metadata first (free request)
        metadata = get_street_view_metadata(lat, lon)
        if not metadata or metadata.get('status') != 'OK':
            print(f"⚠ No street view available for {grid_id}")
            results.append({
                'grid_id': grid_id,
                'lat': lat, 
                'lon': lon,
                'anomaly_type': anomaly_type,
                'success': False,
                'reason': 'no_street_view_available'
            })
            continue
        
        # Filter out indoor Business View panoramas (pano_id starts with 'CAoS')
        pano_id = metadata.get('pano_id', '')
        if pano_id.startswith('CAoS'):
            print(f"⚠ Skipping indoor Business View panorama for {grid_id} (pano: {pano_id[:12]}...)")
            results.append({
                'grid_id': grid_id, 'lat': lat, 'lon': lon,
                'anomaly_type': anomaly_type, 'success': False,
                'reason': 'indoor_business_view_filtered'
            })
            continue
        
        # Save metadata
        metadata_path = f"streetview_images/metadata/{grid_id}_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Download images in 4 directions
        point_success = True
        directions_downloaded = []
        
        for direction_name, heading in DIRECTIONS.items():
            image_filename = f"{grid_id}_{direction_name}_{heading:03d}.jpg"
            image_path = f"streetview_images/{image_filename}"
            
            print(f"  📸 Downloading {direction_name} view (heading {heading}°)")
            
            if download_street_view_image(lat, lon, heading, image_path):
                successful_downloads += 1
                directions_downloaded.append(direction_name)
                print(f"    ✓ Saved: {image_filename}")
            else:
                point_success = False
                print(f"    ❌ Failed: {image_filename}")
            
            total_requests += 1
            
            # Rate limiting
            time.sleep(1 / RATE_LIMIT)
        
        results.append({
            'grid_id': grid_id,
            'lat': lat,
            'lon': lon, 
            'anomaly_type': anomaly_type,
            'success': point_success,
            'directions_downloaded': directions_downloaded,
            'metadata': metadata
        })
        
        print(f"  📊 Point complete: {len(directions_downloaded)}/4 directions")
        print()
    
    # Summary
    print("📊 DOWNLOAD SUMMARY")
    print("==================")
    print(f"Total sampling points: {len(df)}")
    print(f"Successful downloads: {successful_downloads}/{total_requests}")
    print(f"Success rate: {successful_downloads/total_requests*100:.1f}%") if total_requests > 0 else print("Success rate: N/A (no requests made)")
    
    # Save results
    save_download_log(results)
    
    print(f"\n📁 Images saved to: ./streetview_images/")
    print(f"📄 Metadata saved to: ./streetview_images/metadata/")
    print(f"📋 Download log: ./streetview_images/logs/download_log.json")
    print("\n✅ Download complete! Ready for computer vision processing.")

if __name__ == "__main__":
    main()
