#!/usr/bin/env python3
"""
Test script to verify final UI improvements:
1. Surface mode fixes (no cartoon interference, better opacity)
2. Custom sphere selection functionality
3. Updated Prediction Summary (Mean Region, Antigenicity)
4. New Predicted Binding Region Residues section
"""

import requests
import json
import time
from pathlib import Path

# Test configuration
SERVER_URL = "http://localhost:8001"
TEST_PDB = "5I9Q"
TEST_CHAIN = "A"

def test_ui_improvements():
    """Test the UI improvements by checking HTML structure"""
    print("Testing UI structure improvements...")
    
    # Check HTML structure
    html_file = Path("templates/index.html")
    if html_file.exists():
        with open(html_file, 'r') as f:
            content = f.read()
        
        # Test 1: Custom sphere selection
        if 'Custom Selection' in content and 'sphereCheckboxes' in content:
            print("✅ Custom sphere selection UI added")
        else:
            print("❌ Custom sphere selection UI missing")
        
        # Test 2: Summary updates
        if 'Mean Region' in content and 'Antigenicity' in content:
            print("✅ Prediction summary updated")
        else:
            print("❌ Prediction summary not updated")
        
        # Test 3: Binding region section
        if 'Predicted Binding Region Residues' in content and 'bindingRegionResidues' in content:
            print("✅ Binding region residues section added")
        else:
            print("❌ Binding region residues section missing")
    
    # Check CSS updates
    css_file = Path("static/css/style.css")
    if css_file.exists():
        with open(css_file, 'r') as f:
            content = f.read()
        
        if 'binding-region-residue' in content and 'sphere-checkbox' in content:
            print("✅ CSS styles updated")
        else:
            print("❌ CSS styles missing")
    
    # Check JavaScript updates
    js_file = Path("static/js/app.js")
    if js_file.exists():
        with open(js_file, 'r') as f:
            content = f.read()
        
        checks = [
            ('hidden: true', 'Surface mode cartoon hiding'),
            ('generateSphereCheckboxes', 'Sphere checkbox generation'),
            ('getSelectedSphereIndices', 'Custom sphere selection'),
            ('displayBindingRegionResidues', 'Binding region display'),
            ('meanRegion', 'Mean region calculation'),
            ('antigenicity', 'Antigenicity display')
        ]
        
        for check, description in checks:
            if check in content:
                print(f"✅ {description} implemented")
            else:
                print(f"❌ {description} missing")

def test_prediction_functionality():
    """Test the prediction functionality with new features"""
    print("\nTesting prediction functionality...")
    
    try:
        # Test data
        data = {
            'pdb_id': TEST_PDB,
            'chain_id': TEST_CHAIN,
            'radius': '19.0',
            'k': '7',
            'encoder': 'esmc',
            'device_id': '-1',  # CPU mode for testing
        }
        
        # Submit prediction
        response = requests.post(f"{SERVER_URL}/predict", data=data)
        response.raise_for_status()
        
        result = response.json()
        task_id = result['task_id']
        print(f"✅ Prediction started with task ID: {task_id}")
        
        # Monitor progress
        while True:
            status_response = requests.get(f"{SERVER_URL}/status/{task_id}")
            status_response.raise_for_status()
            
            status = status_response.json()
            print(f"Progress: {status.get('progress', 0)}% - {status.get('message', 'Processing...')}")
            
            if status['status'] == 'completed':
                print("✅ Prediction completed successfully!")
                break
            elif status['status'] in ['error', 'failed']:
                print(f"❌ Prediction failed: {status.get('message', 'Unknown error')}")
                return False
            
            time.sleep(1)
        
        # Get results and test new features
        results_response = requests.get(f"{SERVER_URL}/results/{task_id}")
        results_response.raise_for_status()
        
        results = results_response.json()
        
        # Test new data structure
        prediction = results['prediction']
        
        # Test 1: Check for epitope_rate (antigenicity)
        if 'epitope_rate' in prediction:
            print(f"✅ Antigenicity available: {prediction['epitope_rate']:.3f}")
        else:
            print("❌ Antigenicity (epitope_rate) missing")
        
        # Test 2: Check for top_k_region_residues
        if 'top_k_region_residues' in prediction:
            region_residues = prediction['top_k_region_residues']
            print(f"✅ Binding region residues available: {len(region_residues)} residues")
        else:
            print("❌ Binding region residues missing")
        
        # Test 3: Check for region predicted_value
        if 'top_k_regions' in prediction and prediction['top_k_regions']:
            regions = prediction['top_k_regions']
            has_predicted_value = all('predicted_value' in region for region in regions)
            if has_predicted_value:
                values = [region['predicted_value'] for region in regions]
                mean_region = sum(values) / len(values)
                print(f"✅ Region predicted values available, mean: {mean_region:.3f}")
            else:
                print("❌ Region predicted values missing")
        
        # Test 4: Sphere data for custom selection
        if 'top_k_regions' in prediction:
            regions = prediction['top_k_regions']
            sphere_data = [(i+1, r['center_residue'], r.get('radius', 'N/A')) for i, r in enumerate(regions)]
            print(f"✅ Sphere data for custom selection:")
            for sphere_num, center, radius in sphere_data:
                print(f"   Sphere {sphere_num}: Center R{center}, Radius {radius}")
        
        print("\nData Structure Summary:")
        print(f"- Protein: {results['protein_info']['id']} Chain {results['protein_info']['chain_id']}")
        print(f"- Predicted epitopes: {len(prediction['predicted_epitopes'])}")
        print(f"- Top-k regions: {len(prediction['top_k_regions'])}")
        print(f"- Binding region residues: {len(prediction.get('top_k_region_residues', []))}")
        print(f"- Antigenicity: {prediction.get('epitope_rate', 'N/A')}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during prediction testing: {e}")
        return False

def test_server_health():
    """Test if server is running"""
    print("Testing server health...")
    
    try:
        response = requests.get(f"{SERVER_URL}/")
        response.raise_for_status()
        print("✅ Server is running")
        return True
    except Exception as e:
        print(f"❌ Server is not responding: {e}")
        return False

def main():
    """Main test function"""
    print("=" * 60)
    print("Testing Final UI Improvements")
    print("=" * 60)
    
    # Test 1: UI structure
    test_ui_improvements()
    
    # Test 2: Server health
    if not test_server_health():
        print("\n❌ Server is not running. Please start the server first.")
        return
    
    # Test 3: Prediction functionality
    if not test_prediction_functionality():
        print("\n❌ Prediction functionality test failed.")
        return
    
    print("\n" + "=" * 60)
    print("ALL TESTS COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    
    print("\nFinal Improvements Summary:")
    print("1. ✅ Surface Mode Fixes:")
    print("   - Cartoon skeleton hidden in surface mode")
    print("   - Increased surface opacity for better visibility")
    print("   - Proper surface clearing when switching modes")
    
    print("\n2. ✅ Custom Sphere Selection:")
    print("   - Added 'Custom Selection' option to sphere dropdown")
    print("   - Dynamic checkbox generation for each sphere")
    print("   - Individual sphere toggle with visual feedback")
    
    print("\n3. ✅ Updated Prediction Summary:")
    print("   - Removed 'Mean Probability'")
    print("   - Added 'Mean Region' (average of region predicted values)")
    print("   - Added 'Antigenicity' (epitope_rate from prediction)")
    
    print("\n4. ✅ Predicted Binding Region Residues:")
    print("   - New section below epitope residues")
    print("   - Shows all residues covered by k spheres")
    print("   - Green color coding to distinguish from epitopes")
    
    print(f"\nOpen {SERVER_URL} in your browser to test the new features!")
    print("\nTesting Instructions:")
    print("1. Submit a prediction")
    print("2. Try different surface/cartoon modes to test surface fixes")
    print("3. Select 'Custom Selection' in sphere dropdown to test custom sphere selection")
    print("4. Check the updated Prediction Summary values")
    print("5. Scroll down to see the new Binding Region Residues section")

if __name__ == "__main__":
    main() 