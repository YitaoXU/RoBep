#!/usr/bin/env python3
import requests
import json
import time

# Test the API endpoints
BASE_URL = "http://localhost:8000"

def test_prediction_api():
    print("Testing BCE Prediction API...")
    
    # Test health check
    print("\n1. Testing health check...")
    try:
        response = requests.get(f"{BASE_URL}/health")
        print(f"Status: {response.status_code}")
        print(f"Response: {response.json()}")
    except Exception as e:
        print(f"Health check failed: {e}")
        return
    
    # Test prediction with PDB ID
    print("\n2. Testing prediction with PDB ID...")
    try:
        # Submit prediction
        data = {
            'pdb_id': '5I9Q',
            'chain_id': 'A',
            'device_id': '-1',  # Use CPU for testing
            'radius': '19.0',
            'k': '7',
            'encoder': 'esmc'
        }
        
        response = requests.post(f"{BASE_URL}/predict", data=data)
        print(f"Submit status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            task_id = result['task_id']
            print(f"Task ID: {task_id}")
            
            # Monitor progress
            print("\n3. Monitoring progress...")
            for i in range(30):  # Wait up to 60 seconds
                try:
                    status_response = requests.get(f"{BASE_URL}/status/{task_id}")
                    if status_response.status_code == 200:
                        status = status_response.json()
                        print(f"Progress: {status['progress']}% - {status['message']}")
                        
                        if status['status'] == 'completed':
                            print("\n4. Getting results...")
                            result_response = requests.get(f"{BASE_URL}/result/{task_id}")
                            if result_response.status_code == 200:
                                result_data = result_response.json()
                                print(f"Success! Found {len(result_data['prediction']['predicted_epitopes'])} epitope residues")
                                print(f"Protein: {result_data['protein_info']['id']} Chain {result_data['protein_info']['chain_id']}")
                                print(f"Length: {result_data['protein_info']['length']} residues")
                                return True
                            else:
                                print(f"Failed to get results: {result_response.status_code}")
                                return False
                        elif status['status'] == 'error':
                            print(f"Prediction failed: {status['error']}")
                            return False
                    else:
                        print(f"Status check failed: {status_response.status_code}")
                        return False
                except Exception as e:
                    print(f"Status check error: {e}")
                    return False
                
                time.sleep(2)  # Wait 2 seconds before checking again
            
            print("Timeout waiting for prediction to complete")
            return False
        else:
            print(f"Failed to submit prediction: {response.status_code}")
            print(f"Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"API test failed: {e}")
        return False

if __name__ == "__main__":
    success = test_prediction_api()
    if success:
        print("\n✅ API test passed!")
    else:
        print("\n❌ API test failed!") 