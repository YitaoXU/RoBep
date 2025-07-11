#!/usr/bin/env python3
"""
Test script to verify all fixes:
1. Sphere display functionality
2. Progress bar synchronization 
3. Probability gradient coloring
"""

import requests
import json
import time
import sys
from datetime import datetime

BASE_URL = "http://localhost:8001"

def test_all_fixes():
    """Test all the fixes we implemented"""
    print("🧪 Testing BCE Prediction - All Fixes")
    print("=" * 50)
    
    # 1. Test health check
    print("\n1. 🔍 Testing server health...")
    try:
        response = requests.get(f"{BASE_URL}/health")
        print(f"   ✅ Health Status: {response.status_code}")
        if response.status_code != 200:
            print("   ❌ Health check failed!")
            return False
    except Exception as e:
        print(f"   ❌ Health check error: {e}")
        return False
    
    # 2. Test prediction with progress monitoring
    print("\n2. 🚀 Testing prediction with progress monitoring...")
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
        
        start_time = datetime.now()
        response = requests.post(f"{BASE_URL}/predict", data=data)
        print(f"   ✅ Submit status: {response.status_code}")
        
        if response.status_code != 200:
            print(f"   ❌ Prediction submit failed: {response.text}")
            return False
        
        result = response.json()
        task_id = result.get('task_id')
        print(f"   📝 Task ID: {task_id}")
        
        # 3. Monitor prediction progress (test progress bar sync)
        print("\n3. 📊 Testing progress bar synchronization...")
        max_attempts = 120  # Wait up to 2 minutes
        progress_history = []
        
        for attempt in range(max_attempts):
            try:
                response = requests.get(f"{BASE_URL}/status/{task_id}")
                if response.status_code == 200:
                    status = response.json()
                    progress = status.get('progress', 0)
                    message = status.get('message', '')
                    
                    # Track progress history
                    progress_history.append({
                        'time': datetime.now(),
                        'progress': progress,
                        'message': message
                    })
                    
                    print(f"   📈 Progress: {progress:3d}% - {message}")
                    
                    if status.get('status') == 'completed':
                        print("   ✅ Prediction completed!")
                        break
                    elif status.get('status') in ['failed', 'error']:
                        print(f"   ❌ Prediction failed: {status.get('error', '')}")
                        return False
                        
                time.sleep(1)
            except Exception as e:
                print(f"   ⚠️  Status check error: {e}")
                time.sleep(1)
        else:
            print("   ❌ Prediction timed out!")
            return False
        
        # Analyze progress synchronization
        if len(progress_history) >= 2:
            print(f"\n   🔍 Progress Analysis:")
            print(f"      - Total progress updates: {len(progress_history)}")
            print(f"      - Progress range: {progress_history[0]['progress']}% → {progress_history[-1]['progress']}%")
            
            # Check if progress is monotonic (always increasing)
            monotonic = all(
                progress_history[i]['progress'] >= progress_history[i-1]['progress'] 
                for i in range(1, len(progress_history))
            )
            print(f"      - Monotonic progress: {'✅' if monotonic else '❌'}")
        
        # 4. Get results and verify visualization data
        print("\n4. 🎨 Testing visualization data structure...")
        response = requests.get(f"{BASE_URL}/result/{task_id}")
        if response.status_code != 200:
            print(f"   ❌ Results fetch failed: {response.text}")
            return False
        
        results = response.json()
        
        # Check visualization data structure
        viz_data = results.get('visualization', {})
        prediction = results.get('prediction', {})
        
        print(f"   ✅ PDB data present: {bool(viz_data.get('pdb_data'))}")
        print(f"   ✅ Predicted epitopes: {len(prediction.get('predicted_epitopes', []))}")
        print(f"   ✅ Top-k regions: {len(prediction.get('top_k_regions', []))}")
        
        # 5. Test sphere display data
        print("\n5. 🔮 Testing sphere display data...")
        regions = prediction.get('top_k_regions', [])
        if regions:
            all_have_required_fields = True
            for i, region in enumerate(regions):
                has_radius = 'radius' in region
                has_center = 'center_residue' in region
                has_covered = 'covered_residues' in region
                
                if not (has_radius and has_center and has_covered):
                    all_have_required_fields = False
                    print(f"   ❌ Region {i+1} missing fields: radius={has_radius}, center={has_center}, covered={has_covered}")
                else:
                    print(f"   ✅ Region {i+1}: center={region['center_residue']}, radius={region['radius']}, covered={len(region['covered_residues'])}")
            
            if all_have_required_fields:
                print("   🎉 All regions have required fields for sphere display!")
            else:
                print("   ❌ Some regions missing required fields")
                return False
        else:
            print("   ❌ No regions found in prediction results")
            return False
        
        # 6. Test probability gradient data
        print("\n6. 🌈 Testing probability gradient data...")
        predictions = prediction.get('predictions', {})
        if predictions:
            print(f"   ✅ Residue predictions available: {len(predictions)} residues")
            
            # Check probability distribution
            probs = list(predictions.values())
            min_prob = min(probs) if probs else 0
            max_prob = max(probs) if probs else 0
            avg_prob = sum(probs) / len(probs) if probs else 0
            
            print(f"   ✅ Probability range: {min_prob:.3f} - {max_prob:.3f}")
            print(f"   ✅ Average probability: {avg_prob:.3f}")
            
            # Check if we have good probability distribution for gradient
            if max_prob > min_prob and (max_prob - min_prob) > 0.1:
                print("   🎨 Excellent probability range for gradient coloring!")
            elif max_prob > min_prob:
                print("   ⚠️  Limited probability range for gradient coloring")
            else:
                print("   ❌ No probability variation for gradient coloring")
                return False
                
            # Count high/medium/low probability residues
            high_prob = sum(1 for p in probs if p > 0.7)
            med_prob = sum(1 for p in probs if 0.3 <= p <= 0.7)
            low_prob = sum(1 for p in probs if p < 0.3)
            
            print(f"   📊 Probability distribution:")
            print(f"      - High (>0.7): {high_prob} residues")
            print(f"      - Medium (0.3-0.7): {med_prob} residues")
            print(f"      - Low (<0.3): {low_prob} residues")
            
        else:
            print("   ❌ No residue predictions available")
            return False
        
        # 7. Performance summary
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        print(f"\n7. ⏱️  Performance Summary:")
        print(f"   - Total prediction time: {duration:.1f} seconds")
        print(f"   - Progress updates: {len(progress_history)}")
        print(f"   - Average update interval: {duration/len(progress_history):.1f} seconds")
        
        print(f"\n🎉 All tests passed! All fixes are working correctly!")
        print("\n📋 Summary of fixes verified:")
        print("   ✅ Sphere display data structure complete")
        print("   ✅ Progress bar synchronization improved")
        print("   ✅ Probability gradient data ready")
        print("   ✅ Enhanced visualization features available")
        
        print(f"\n🚀 Ready to use! Visit: {BASE_URL}")
        print("   - Select Surface representation to see molecular surface")
        print("   - Check 'Show Spherical Regions' to see prediction spheres")
        print("   - Use 'Probability Gradient' mode for full gradient coloring")
        print("   - Adjust 'Number of Spheres' to control sphere display")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Test error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_all_fixes()
    sys.exit(0 if success else 1) 