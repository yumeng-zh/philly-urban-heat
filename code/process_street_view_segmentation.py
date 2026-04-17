#!/usr/bin/env python3
"""
SegFormer Street View Semantic Segmentation
==========================================

Processes street view images to extract urban morphology indicators:
- Green View Index (GVI) - vegetation pixel ratio
- Sky View Factor (SVF) - sky pixel ratio  
- Building View Factor (BVF) - building pixel ratio
- Road View Factor (RVF) - road pixel ratio
+ Derived indicators: Canyon Ratio, Canopy Height Proxy

Uses SegFormer-B0 model with ADE20K weights for semantic segmentation.

Usage:
    pip install transformers torch pillow numpy pandas --break-system-packages
    python process_street_view_segmentation.py

Input: ./streetview_images/*.jpg
Output: ./cv_results/ with indicator calculations and visualizations
"""

import os
import glob
import json
import numpy as np
import pandas as pd
from PIL import Image
import torch
from datetime import datetime

try:
    from transformers import SegformerFeatureExtractor, SegformerForSemanticSegmentation
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    print("⚠ Transformers not available. Install with:")
    print("  pip install transformers torch --break-system-packages")
    TRANSFORMERS_AVAILABLE = False

# =============================================================================
# SEGMENTATION CLASSES (ADE20K dataset mapping)
# =============================================================================

# Key classes for urban morphology analysis
URBAN_CLASSES = {
    'sky': [2],                    # Sky 
    'vegetation': [4, 17, 61],     # Tree, Plant, Grass
    'building': [0, 25, 6],        # Wall, House, Building
    'road': [6, 7, 11],            # Road, Pavement, Path
    'other': []                    # Everything else
}

# =============================================================================
# CORE FUNCTIONS  
# =============================================================================

def setup_segformer_model():
    """Load SegFormer model and feature extractor"""
    if not TRANSFORMERS_AVAILABLE:
        return None, None
        
    print("🤖 Loading SegFormer-B0 model...")
    try:
        feature_extractor = SegformerFeatureExtractor.from_pretrained(
            "nvidia/segformer-b0-finetuned-ade-512-512"
        )
        model = SegformerForSemanticSegmentation.from_pretrained(
            "nvidia/segformer-b0-finetuned-ade-512-512"
        )
        print("✅ SegFormer model loaded successfully")
        return feature_extractor, model
    except Exception as e:
        print(f"❌ Failed to load SegFormer model: {e}")
        return None, None

def segment_image(image_path, feature_extractor, model):
    """
    Perform semantic segmentation on street view image
    Returns: segmentation mask as numpy array
    """
    try:
        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        inputs = feature_extractor(images=image, return_tensors="pt")
        
        # Run inference
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
        
        # Get segmentation mask
        predicted_segmentation = torch.nn.functional.interpolate(
            logits,
            size=image.size[::-1],  # (height, width)
            mode="bilinear",
            align_corners=False,
        ).argmax(dim=1)
        
        return predicted_segmentation[0].cpu().numpy()
    
    except Exception as e:
        print(f"⚠ Segmentation failed for {image_path}: {e}")
        return None

def calculate_class_masks(segmentation_mask):
    """
    Convert segmentation to binary masks for each urban class
    Returns: dict with class name -> binary mask
    """
    masks = {}
    
    for class_name, class_ids in URBAN_CLASSES.items():
        if class_ids:  # Skip 'other' class for now
            mask = np.zeros_like(segmentation_mask, dtype=bool)
            for class_id in class_ids:
                mask |= (segmentation_mask == class_id)
            masks[class_name] = mask
    
    # Handle 'other' class as everything not in main classes
    main_mask = np.zeros_like(segmentation_mask, dtype=bool)
    for class_name in ['sky', 'vegetation', 'building', 'road']:
        if class_name in masks:
            main_mask |= masks[class_name]
    masks['other'] = ~main_mask
    
    return masks

def calculate_morphology_indicators(masks, image_height):
    """
    Calculate urban morphology indicators from class masks
    """
    total_pixels = masks['sky'].size
    
    # Primary indicators (pixel ratios)
    gvi = np.sum(masks['vegetation']) / total_pixels  # Green View Index
    svf = np.sum(masks['sky']) / total_pixels         # Sky View Factor  
    bvf = np.sum(masks['building']) / total_pixels    # Building View Factor
    rvf = np.sum(masks['road']) / total_pixels        # Road View Factor
    
    # Derived indicators
    canyon_ratio = bvf / svf if svf > 0 else 0  # Canyon depth proxy
    
    # Canopy Height Proxy: vegetation in upper 1/3 vs total vegetation
    upper_third = image_height // 3
    upper_mask = np.zeros_like(masks['vegetation'])
    upper_mask[:upper_third, :] = True
    
    upper_vegetation = np.sum(masks['vegetation'] & upper_mask)
    total_vegetation = np.sum(masks['vegetation'])
    canopy_height_proxy = upper_vegetation / total_vegetation if total_vegetation > 0 else 0
    
    # Vegetation-to-sky ratio
    veg_sky_ratio = gvi / svf if svf > 0 else 0
    
    return {
        'gvi': gvi,
        'svf': svf, 
        'bvf': bvf,
        'rvf': rvf,
        'canyon_ratio': canyon_ratio,
        'canopy_height_proxy': canopy_height_proxy,
        'vegetation_sky_ratio': veg_sky_ratio,
        'total_pixels': total_pixels
    }

def save_visualization(image_path, segmentation_mask, masks, output_dir):
    """Save segmentation visualization"""
    try:
        # Create color-coded segmentation overlay
        height, width = segmentation_mask.shape
        colored_mask = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Color mapping
        colors = {
            'vegetation': [34, 139, 34],   # Forest Green
            'sky': [135, 206, 235],        # Sky Blue  
            'building': [169, 169, 169],   # Dark Gray
            'road': [105, 105, 105],       # Dim Gray
            'other': [255, 255, 255]       # White
        }
        
        for class_name, color in colors.items():
            colored_mask[masks[class_name]] = color
        
        # Save visualization
        filename = os.path.basename(image_path).replace('.jpg', '_segmentation.png')
        output_path = os.path.join(output_dir, filename)
        
        Image.fromarray(colored_mask).save(output_path)
        return output_path
        
    except Exception as e:
        print(f"⚠ Visualization save failed: {e}")
        return None

def find_street_view_images():
    """Find all street view images to process"""
    image_patterns = [
        'streetview_images/*.jpg',
        './streetview_images/*.jpg', 
        '*.jpg'  # Current directory fallback
    ]
    
    images = []
    for pattern in image_patterns:
        found = glob.glob(pattern)
        if found:
            images.extend(found)
            print(f"✓ Found {len(found)} images matching {pattern}")
            break
    
    if not images:
        print("⚠ No street view images found. Expected in ./streetview_images/")
        # Create sample notification
        print("📋 Expected file structure:")
        print("   ./streetview_images/grid_001_north_000.jpg")
        print("   ./streetview_images/grid_001_east_090.jpg")
        print("   ./streetview_images/grid_001_south_180.jpg")
        print("   ./streetview_images/grid_001_west_270.jpg")
    
    return images

def create_output_directories():
    """Create output directories for results"""
    dirs = ['cv_results', 'cv_results/visualizations', 'cv_results/data']
    for d in dirs:
        os.makedirs(d, exist_ok=True)

def main():
    """Main processing function"""
    print("🔬 SegFormer Street View Processor")
    print("==================================")
    
    # Setup
    create_output_directories()
    images = find_street_view_images()
    
    if not images:
        return
    
    if not TRANSFORMERS_AVAILABLE:
        print("❌ Cannot proceed without transformers library")
        return
    
    # Load model
    feature_extractor, model = setup_segformer_model()
    if feature_extractor is None:
        return
    
    print(f"\n📸 Processing {len(images)} street view images")
    
    # Process images
    results = []
    
    for i, image_path in enumerate(images):
        filename = os.path.basename(image_path)
        print(f"\n🔍 Processing {filename} ({i+1}/{len(images)})")
        
        # Parse filename to extract metadata
        parts = filename.replace('.jpg', '').split('_')
        if len(parts) >= 4:
            grid_id = '_'.join(parts[:-2])  # Everything except direction and heading
            direction = parts[-2]
            heading = parts[-1]
        else:
            grid_id = filename.replace('.jpg', '')
            direction = 'unknown'
            heading = '000'
        
        # Perform segmentation
        segmentation_mask = segment_image(image_path, feature_extractor, model)
        if segmentation_mask is None:
            continue
        
        # Calculate class masks
        masks = calculate_class_masks(segmentation_mask)
        
        # Calculate indicators
        image_height = segmentation_mask.shape[0]
        indicators = calculate_morphology_indicators(masks, image_height)
        
        # Save visualization
        viz_path = save_visualization(
            image_path, segmentation_mask, masks, 'cv_results/visualizations'
        )
        
        # Store results
        result = {
            'grid_id': grid_id,
            'direction': direction,
            'heading': int(heading) if heading.isdigit() else 0,
            'image_path': image_path,
            'visualization_path': viz_path,
            **indicators
        }
        results.append(result)
        
        print(f"  📊 GVI: {indicators['gvi']:.3f} | SVF: {indicators['svf']:.3f}")
        print(f"  📊 BVF: {indicators['bvf']:.3f} | RVF: {indicators['rvf']:.3f}")
    
    # Save results
    if results:
        results_df = pd.DataFrame(results)
        
        # Per-image results
        results_df.to_csv('cv_results/data/per_image_indicators.csv', index=False)
        
        # Grid-level aggregation (mean across 4 directions)
        if 'grid_id' in results_df.columns:
            grid_results = results_df.groupby('grid_id').agg({
                'gvi': 'mean',
                'svf': 'mean', 
                'bvf': 'mean',
                'rvf': 'mean',
                'canyon_ratio': 'mean',
                'canopy_height_proxy': 'mean',
                'vegetation_sky_ratio': 'mean'
            }).round(4)
            
            grid_results.to_csv('cv_results/data/grid_level_indicators.csv')
            
            print(f"\n📊 PROCESSING SUMMARY")
            print(f"===================")
            print(f"Images processed: {len(results)}")
            print(f"Unique grids: {len(grid_results)}")
            print(f"Mean indicators across all grids:")
            print(f"  GVI: {grid_results['gvi'].mean():.3f} ± {grid_results['gvi'].std():.3f}")
            print(f"  SVF: {grid_results['svf'].mean():.3f} ± {grid_results['svf'].std():.3f}")
            print(f"  BVF: {grid_results['bvf'].mean():.3f} ± {grid_results['bvf'].std():.3f}")
            print(f"  RVF: {grid_results['rvf'].mean():.3f} ± {grid_results['rvf'].std():.3f}")
        
        # Save processing log
        log = {
            'timestamp': datetime.now().isoformat(),
            'total_images': len(images),
            'successful_processing': len(results),
            'model_used': 'nvidia/segformer-b0-finetuned-ade-512-512',
            'indicators_calculated': list(results[0].keys()) if results else []
        }
        
        with open('cv_results/processing_log.json', 'w') as f:
            json.dump(log, f, indent=2)
        
        print(f"\n📁 Results saved to:")
        print(f"  Per-image: cv_results/data/per_image_indicators.csv")
        print(f"  Grid-level: cv_results/data/grid_level_indicators.csv")
        print(f"  Visualizations: cv_results/visualizations/")
        print(f"  Processing log: cv_results/processing_log.json")
        
    else:
        print("\n❌ No images were successfully processed")
    
    print("\n✅ Computer vision processing complete!")

if __name__ == "__main__":
    main()
