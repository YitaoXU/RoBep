#!/usr/bin/env python3
"""
Test script to verify surface mode and sphere display fixes
"""

import requests
import json
import time
import sys

BASE_URL = "http://localhost:8001"

def test_visualization_fixes():
    """Test the surface mode and sphere display fixes"""
    print("Testing BCE Prediction Visualization Fixes...")
    
    # 1. Test health check
    print("\n1. Testing health check...")
    try:
        response = requests.get(f"{BASE_URL}/health")
        print(f"Health Status: {response.status_code}")
        if response.status_code != 200:
            print("Health check failed!")
            return False
    except Exception as e:
        print(f"Health check error: {e}")
        return False
    
    # 2. Test prediction with PDB ID
    print("\n2. Starting test prediction...")
    try:
        # Submit prediction
        data = {
            'pdb_id': '5I9Q',
            'chain_id': 'A',
            'device_id': '0',  # Use GPU 0
            'radius': '19.0',
            'k': '7',
            'encoder': 'esmc'
        }
        
        response = requests.post(f"{BASE_URL}/predict", data=data)
        print(f"Submit status: {response.status_code}")
        
        if response.status_code != 200:
            print(f"Prediction submit failed: {response.text}")
            return False
        
        result = response.json()
        task_id = result.get('task_id')
        print(f"Task ID: {task_id}")
        
        # 3. Monitor prediction progress
        print("\n3. Monitoring prediction progress...")
        max_attempts = 60  # Wait up to 5 minutes
        for attempt in range(max_attempts):
            try:
                response = requests.get(f"{BASE_URL}/status/{task_id}")
                if response.status_code == 200:
                    status = response.json()
                    print(f"Progress: {status.get('progress', 0)}% - {status.get('message', '')}")
                    
                    if status.get('status') == 'completed':
                        print("Prediction completed!")
                        break
                    elif status.get('status') == 'failed':
                        print(f"Prediction failed: {status.get('error', '')}")
                        return False
                        
                time.sleep(5)
            except Exception as e:
                print(f"Status check error: {e}")
                time.sleep(5)
        else:
            print("Prediction timed out!")
            return False
        
        # 4. Get results and verify visualization data
        print("\n4. Verifying visualization data...")
        response = requests.get(f"{BASE_URL}/result/{task_id}")
        if response.status_code != 200:
            print(f"Results fetch failed: {response.text}")
            return False
        
        results = response.json()
        
        # Check visualization data structure
        viz_data = results.get('visualization', {})
        prediction = results.get('prediction', {})
        
        print(f"✓ PDB data present: {bool(viz_data.get('pdb_data'))}")
        print(f"✓ Predicted epitopes: {len(prediction.get('predicted_epitopes', []))}")
        print(f"✓ Top-k regions: {len(prediction.get('top_k_regions', []))}")
        
        # Check if top_k_regions have radius information
        regions = prediction.get('top_k_regions', [])
        if regions:
            first_region = regions[0]
            has_radius = 'radius' in first_region
            has_center = 'center_residue' in first_region
            has_covered = 'covered_residues' in first_region
            
            print(f"✓ Region has radius: {has_radius}")
            print(f"✓ Region has center_residue: {has_center}")
            print(f"✓ Region has covered_residues: {has_covered}")
            
            if has_radius and has_center and has_covered:
                print("\n✅ All visualization data structure checks passed!")
                print(f"   - Center residue: {first_region['center_residue']}")
                print(f"   - Radius: {first_region['radius']}")
                print(f"   - Covered residues: {len(first_region['covered_residues'])}")
            else:
                print("\n❌ Missing required fields in region data")
                return False
        else:
            print("❌ No regions found in prediction results")
            return False
        
        # 5. Test surface mode data
        print("\n5. Checking surface mode compatibility...")
        predictions = prediction.get('predictions', {})
        if predictions:
            print(f"✓ Residue predictions available: {len(predictions)} residues")
            
            # Check if we have probability data for surface coloring
            probs = list(predictions.values())
            min_prob = min(probs) if probs else 0
            max_prob = max(probs) if probs else 0
            print(f"✓ Probability range: {min_prob:.3f} - {max_prob:.3f}")
            
            if max_prob > min_prob:
                print("✅ Surface mode probability gradient data is ready!")
            else:
                print("⚠️  Surface mode may have limited color variation")
        else:
            print("❌ No residue predictions available for surface mode")
            return False
        
        print("\n🎉 All visualization tests passed!")
        print("Surface mode and sphere display should now work correctly!")
        return True
        
    except Exception as e:
        print(f"Test error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_visualization_fixes()
    sys.exit(0 if success else 1) 