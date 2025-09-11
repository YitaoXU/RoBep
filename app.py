import gradio as gr
import os
import json
import tempfile
import traceback
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
import torch
import time
import io
import base64
import zipfile
from datetime import datetime

# Dynamic installation of PyTorch Geometric dependencies
def install_torch_geometric_deps():
    """Install PyTorch Geometric dependencies at runtime to avoid compilation issues during Hugging Face Spaces build"""
    import subprocess
    import sys
    
    # Check if torch-scatter is already installed
    try:
        import torch_scatter
        print("✅ torch-scatter already installed")
        return True
    except ImportError:
        print("🔄 Installing torch-scatter and related packages...")
        
        # Get PyTorch version and CUDA info
        torch_version = torch.__version__
        torch_version_str = '+'.join(torch_version.split('+')[:1])  # Remove CUDA info
        
        # Use PyTorch Geometric official recommended installation method
        try:
            # For CPU version, use official CPU wheel
            pip_cmd = [
                sys.executable, "-m", "pip", "install", 
                "torch-scatter", "torch-sparse", "torch-cluster", "torch-spline-conv",
                "-f", f"https://data.pyg.org/whl/torch-{torch_version_str}+cpu.html",
                "--no-cache-dir"
            ]
            
            print(f"Running: {' '.join(pip_cmd)}")
            result = subprocess.run(pip_cmd, capture_output=True, text=True, timeout=300)
            
            if result.returncode == 0:
                print("✅ Successfully installed torch-scatter and related packages")
                return True
            else:
                print(f"❌ Failed to install packages: {result.stderr}")
                # Try simplified installation method
                try:
                    simple_cmd = [sys.executable, "-m", "pip", "install", "torch-scatter", "--no-cache-dir"]
                    result = subprocess.run(simple_cmd, capture_output=True, text=True, timeout=300)
                    if result.returncode == 0:
                        print("✅ Successfully installed torch-scatter with simple method")
                        return True
                    else:
                        print(f"❌ Simple install also failed: {result.stderr}")
                        return False
                except Exception as e:
                    print(f"❌ Exception during simple install: {e}")
                    return False
                    
        except subprocess.TimeoutExpired:
            print("❌ Installation timeout - packages may not be available")
            return False
        except Exception as e:
            print(f"❌ Exception during installation: {e}")
            return False

# Try to install PyTorch Geometric dependencies
deps_installed = install_torch_geometric_deps()

if not deps_installed:
    print("⚠️ Warning: PyTorch Geometric dependencies not installed. Some features may not work.")
    print("The application will try to continue with limited functionality.")

# Set up paths and imports for different deployment environments
import sys
BASE_DIR = Path(__file__).parent

# Smart import handling for different environments
def setup_imports():
    """Smart import setup for different deployment environments"""
    global AntigenChain, PROJECT_BASE_DIR
    
    # Method 1: Try importing from src directory (local development)
    if (BASE_DIR / "src").exists():
        sys.path.insert(0, str(BASE_DIR))
        try:
            from src.bce.antigen.antigen import AntigenChain
            from src.bce.utils.constants import BASE_DIR as PROJECT_BASE_DIR
            print("✅ Successfully imported from src/ directory")
            return True
        except ImportError as e:
            print(f"❌ Failed to import from src/: {e}")
    
    # Method 2: Try adding src to path and direct import (Hugging Face Spaces)
    src_path = BASE_DIR / "src"
    if src_path.exists():
        sys.path.insert(0, str(src_path))
        try:
            from bce.antigen.antigen import AntigenChain
            from bce.utils.constants import BASE_DIR as PROJECT_BASE_DIR
            print("✅ Successfully imported from src/ added to path")
            return True
        except ImportError as e:
            print(f"❌ Failed to import with src/ in path: {e}")
    
    # Method 3: Try direct import (if package is installed)
    try:
        from bce.antigen.antigen import AntigenChain
        from bce.utils.constants import BASE_DIR as PROJECT_BASE_DIR
        print("✅ Successfully imported from installed package")
        return True
    except ImportError as e:
        print(f"❌ Failed to import from installed package: {e}")
    
    # If all methods fail, use default settings
    print("⚠️ All import methods failed, using fallback settings")
    PROJECT_BASE_DIR = BASE_DIR
    return False

# Execute import setup
import_success = setup_imports()

if not import_success:
    print("❌ Critical: Could not import BCE modules. Please check the file structure.")
    print("Expected structure:")
    print("- src/bce/antigen/antigen.py")
    print("- src/bce/utils/constants.py")
    print("- src/bce/model/ReCEP.py")
    print("- src/bce/data/utils.py")
    sys.exit(1)

# Configuration
DEFAULT_MODEL_PATH = os.getenv("BCE_MODEL_PATH", str(PROJECT_BASE_DIR / "models" / "ReCEP" / "20250626_110438" / "best_mcc_model.bin"))
ESM_TOKEN = os.getenv("ESM_TOKEN", "1mzAo8l1uxaU8UfVcGgV7B")

# PDB data directory
PDB_DATA_DIR = PROJECT_BASE_DIR / "data" / "pdb"
PDB_DATA_DIR.mkdir(parents=True, exist_ok=True)

def validate_pdb_id(pdb_id: str) -> bool:
    """Validate PDB ID format"""
    if not pdb_id or len(pdb_id) != 4:
        return False
    return pdb_id.isalnum()

def validate_chain_id(chain_id: str) -> bool:
    """Validate chain ID format"""
    if not chain_id or len(chain_id) != 1:
        return False
    return chain_id.isalnum()

def create_pdb_visualization_html(pdb_data: str, predicted_epitopes: list, 
                                 predictions: dict, protein_id: str, top_k_regions: list = None) -> str:
    """Create HTML with 3Dmol.js visualization compatible with Gradio - enhanced version with more features"""
    
    # Prepare data for JavaScript
    epitope_residues = predicted_epitopes
    
    # Process top_k_regions for visualization
    processed_regions = []
    if top_k_regions:
        for i, region in enumerate(top_k_regions):
            if isinstance(region, dict):
                processed_regions.append({
                    'center_idx': region.get('center_idx', 0),
                    'center_residue': region.get('center_residue', region.get('center_idx', 0)),
                    'covered_residues': region.get('covered_residues', region.get('covered_indices', [])),
                    'radius': 18.0,  # Default radius
                    'predicted_value': region.get('graph_pred', 0.0)
                })
    
    # Create a unique ID for this visualization to avoid conflicts
    import uuid
    viewer_id = f"viewer_{uuid.uuid4().hex[:8]}"
    
    html_content = f"""
    <div style="width: 100%; height: 600px; border: 1px solid #ddd; border-radius: 8px; overflow: hidden;">
        <div style="padding: 10px; background: #f8f9fa; border-bottom: 1px solid #ddd;">
            <h3 style="margin: 0 0 10px 0; color: #333;">3D Structure Visualization - {protein_id}</h3>
            <div style="display: flex; gap: 15px; align-items: center; flex-wrap: wrap;">
                <div>
                    <label style="font-weight: bold; margin-right: 5px;">Display Mode:</label>
                    <select id="vizMode_{viewer_id}" onchange="updateVisualization_{viewer_id}()" style="padding: 4px;">
                        <option value="prediction">Predicted Epitopes</option>
                        <option value="probability">Probability Gradient</option>
                        <option value="regions">Top-k Regions</option>
                    </select>
                </div>
                <div>
                    <label style="font-weight: bold; margin-right: 5px;">Style:</label>
                    <select id="vizStyle_{viewer_id}" onchange="updateVisualization_{viewer_id}()" style="padding: 4px;">
                        <option value="cartoon">Cartoon</option>
                        <option value="surface">Surface</option>
                        <option value="stick">Stick</option>
                        <option value="sphere">Sphere</option>
                    </select>
                </div>
                <div>
                    <label style="font-weight: bold; margin-right: 5px;">
                        <input type="checkbox" id="showSpheres_{viewer_id}" onchange="updateVisualization_{viewer_id}()" style="margin-right: 3px;"> Show Spheres
                    </label>
                </div>
                <div>
                    <label style="font-weight: bold; margin-right: 5px;">Sphere Display:</label>
                    <select id="sphereCount_{viewer_id}" onchange="handleSphereCountChange_{viewer_id}()" style="padding: 4px;">
                        <option value="1">Top 1</option>
                        <option value="2">Top 2</option>
                        <option value="3">Top 3</option>
                        <option value="4">Top 4</option>
                        <option value="5" selected>Top 5</option>
                        <option value="6">Top 6</option>
                        <option value="7">Top 7</option>
                        <option value="all">All Spheres</option>
                        <option value="custom">Custom Selection</option>
                    </select>
                </div>
                <div id="customSphereSelection_{viewer_id}" style="display: none; margin-top: 10px; padding: 10px; background: #f9f9f9; border-radius: 5px; max-height: 120px; overflow-y: auto;">
                    <label style="font-weight: bold; margin-bottom: 5px; display: block;">Select Spheres to Display:</label>
                    <div id="sphereCheckboxes_{viewer_id}" style="display: flex; flex-wrap: wrap; gap: 8px; max-height: 80px; overflow-y: auto;">
                        <!-- Checkboxes will be dynamically generated -->
                    </div>
                </div>
                <div>
                    <button onclick="resetView_{viewer_id}()" style="padding: 4px 8px; margin-right: 5px;">Reset View</button>
                    <button onclick="saveImage_{viewer_id}()" style="padding: 4px 8px;">Save Image</button>
                </div>
            </div>
        </div>
        <div id="{viewer_id}" style="width: 100%; height: 520px; min-height: 400px; position: relative; background: #f0f0f0;">
            <div style="position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%); text-align: center;">
                <p id="status_{viewer_id}" style="color: #666;">Loading 3Dmol.js...</p>
            </div>
        </div>
    </div>
    
    <script src="https://unpkg.com/3dmol@2.0.4/build/3Dmol-min.js"></script>
    <script>
        // Global variables for this viewer instance
        window.viewer_{viewer_id} = null;
        window.pdbData_{viewer_id} = `{pdb_data}`;
        window.predictedEpitopes_{viewer_id} = {json.dumps(epitope_residues)};
        window.predictions_{viewer_id} = {json.dumps(predictions)};
        window.topKRegions_{viewer_id} = {json.dumps(processed_regions)};
        
        // Wait for 3Dmol to be available with timeout
        function wait3Dmol_{viewer_id}(attempts = 0) {{
            if (typeof $3Dmol !== 'undefined') {{
                console.log('3Dmol.js loaded successfully for {viewer_id}');
                document.getElementById('status_{viewer_id}').textContent = 'Initializing 3D viewer...';
                setTimeout(() => initializeViewer_{viewer_id}(), 100);
            }} else if (attempts < 50) {{ // 5 second timeout
                console.log(`Waiting for 3Dmol.js... attempt ${{attempts + 1}}`);
                setTimeout(() => wait3Dmol_{viewer_id}(attempts + 1), 100);
            }} else {{
                console.error('Failed to load 3Dmol.js after 5 seconds');
                document.getElementById('status_{viewer_id}').textContent = 'Failed to load 3Dmol.js. Please refresh the page.';
                document.getElementById('status_{viewer_id}').style.color = 'red';
            }}
        }}
        
        function initializeViewer_{viewer_id}() {{
            try {{
                const element = document.getElementById('{viewer_id}');
                if (!element) {{
                    console.error('Viewer element not found: {viewer_id}');
                    return;
                }}
                
                document.getElementById('status_{viewer_id}').textContent = 'Creating viewer...';
                
                window.viewer_{viewer_id} = $3Dmol.createViewer(element, {{
                    defaultcolors: $3Dmol.rasmolElementColors
                }});
                
                document.getElementById('status_{viewer_id}').textContent = 'Loading structure...';
                
                window.viewer_{viewer_id}.addModel(window.pdbData_{viewer_id}, 'pdb');
                
                // Hide status message
                const statusEl = document.getElementById('status_{viewer_id}');
                if (statusEl) statusEl.style.display = 'none';
                
                updateVisualization_{viewer_id}();
                
                // Initialize sphere checkboxes if data is available
                if (window.topKRegions_{viewer_id} && window.topKRegions_{viewer_id}.length > 0) {{
                    generateSphereCheckboxes_{viewer_id}();
                }}
                
                console.log('3D viewer initialized successfully for {viewer_id}');
            }} catch (error) {{
                console.error('Error initializing 3D viewer:', error);
                const statusEl = document.getElementById('status_{viewer_id}');
                if (statusEl) {{
                    statusEl.textContent = 'Error loading 3D viewer: ' + error.message;
                    statusEl.style.color = 'red';
                }}
            }}
        }}
        
        function updateVisualization_{viewer_id}() {{
            if (!window.viewer_{viewer_id}) return;
            
            try {{
                const mode = document.getElementById('vizMode_{viewer_id}').value;
                const style = document.getElementById('vizStyle_{viewer_id}').value;
                const showSpheres = document.getElementById('showSpheres_{viewer_id}').checked;
                
                // Clear everything
                window.viewer_{viewer_id}.removeAllShapes();
                window.viewer_{viewer_id}.removeAllSurfaces();
                window.viewer_{viewer_id}.setStyle({{}}, {{}});
                
                // Base style
                const baseStyle = {{}};
                if (style === 'surface') {{
                    baseStyle['cartoon'] = {{ hidden: true }};
                }} else {{
                    baseStyle[style] = {{ color: '#e6e6f7' }};
                }}
                window.viewer_{viewer_id}.setStyle({{}}, baseStyle);
                
                if (mode === 'prediction') {{
                    // Highlight predicted epitopes
                    if (window.predictedEpitopes_{viewer_id}.length > 0 && style !== 'surface') {{
                        const epitopeStyle = {{}};
                        epitopeStyle[style] = {{ color: '#9C6ADE' }};
                        window.viewer_{viewer_id}.setStyle({{ resi: window.predictedEpitopes_{viewer_id} }}, epitopeStyle);
                    }}
                    
                    // Add surface for epitopes if surface mode
                    if (style === 'surface') {{
                        window.viewer_{viewer_id}.addSurface($3Dmol.SurfaceType.VDW, {{
                            opacity: 1.0,
                            color: '#e6e6f7'
                        }});
                        
                        if (window.predictedEpitopes_{viewer_id}.length > 0) {{
                            window.viewer_{viewer_id}.addSurface($3Dmol.SurfaceType.VDW, {{
                                opacity: 1.0,
                                color: '#9C6ADE'
                            }}, {{ resi: window.predictedEpitopes_{viewer_id} }});
                        }}
                    }}
                }} else if (mode === 'probability') {{
                    // Color by probability scores
                    if (window.predictions_{viewer_id} && Object.keys(window.predictions_{viewer_id}).length > 0) {{
                        const allProbs = Object.values(window.predictions_{viewer_id}).filter(p => p !== undefined);
                        const minProb = Math.min(...allProbs);
                        const maxProb = Math.max(...allProbs);
                        
                        Object.entries(window.predictions_{viewer_id}).forEach(([resnum, score]) => {{
                            const normalizedProb = maxProb > minProb ? (score - minProb) / (maxProb - minProb) : 0.5;
                            const color = interpolateColor('#E6F3FF', '#DC143C', normalizedProb);
                            const probStyle = {{}};
                            if (style !== 'surface') {{
                                probStyle[style] = {{ color: color }};
                                window.viewer_{viewer_id}.setStyle({{ resi: parseInt(resnum) }}, probStyle);
                            }}
                        }});
                        
                        if (style === 'surface') {{
                            window.viewer_{viewer_id}.addSurface($3Dmol.SurfaceType.VDW, {{
                                opacity: 0.8,
                                color: '#e6e6f7'
                            }});
                            
                            Object.entries(window.predictions_{viewer_id}).forEach(([resnum, score]) => {{
                                const normalizedProb = maxProb > minProb ? (score - minProb) / (maxProb - minProb) : 0.5;
                                const color = interpolateColor('#E6F3FF', '#DC143C', normalizedProb);
                                window.viewer_{viewer_id}.addSurface($3Dmol.SurfaceType.VDW, {{
                                    opacity: 1.0,
                                    color: color
                                }}, {{ resi: parseInt(resnum) }});
                            }});
                        }}
                    }}
                }} else if (mode === 'regions') {{
                    // Color top-k regions
                    const colors = ['#FF6B6B', '#96CEB4', '#4ECDC4', '#45B7D1', '#FFEAA7', '#DDA0DD', '#87CEEB'];
                    
                    if (window.topKRegions_{viewer_id} && window.topKRegions_{viewer_id}.length > 0) {{
                        window.topKRegions_{viewer_id}.forEach((region, index) => {{
                            const color = colors[index % colors.length];
                            const regionStyle = {{}};
                            if (style !== 'surface') {{
                                regionStyle[style] = {{ color: color }};
                                window.viewer_{viewer_id}.setStyle({{ resi: region.covered_residues }}, regionStyle);
                            }}
                        }});
                        
                        if (style === 'surface') {{
                            window.viewer_{viewer_id}.addSurface($3Dmol.SurfaceType.VDW, {{
                                opacity: 0.8,
                                color: '#e6e6f7'
                            }});
                            
                            window.topKRegions_{viewer_id}.forEach((region, index) => {{
                                const color = colors[index % colors.length];
                                window.viewer_{viewer_id}.addSurface($3Dmol.SurfaceType.VDW, {{
                                    opacity: 1.0,
                                    color: color
                                }}, {{ resi: region.covered_residues }});
                            }});
                        }}
                    }}
                }}
                
                // Add spheres if requested
                if (showSpheres && window.topKRegions_{viewer_id} && window.topKRegions_{viewer_id}.length > 0) {{
                    const colors = ['#FF6B6B', '#96CEB4', '#4ECDC4', '#45B7D1', '#FFEAA7', '#DDA0DD', '#87CEEB'];
                    const sphereCount = document.getElementById('sphereCount_{viewer_id}').value;
                    
                    // Determine which spheres to show
                    let spheresToShow = [];
                    if (sphereCount === 'custom') {{
                        const selectedIndices = getSelectedSphereIndices_{viewer_id}();
                        spheresToShow = selectedIndices.map(idx => ({{ region: window.topKRegions_{viewer_id}[idx], index: idx }}));
                    }} else {{
                        let numSpheres = sphereCount === 'all' ? window.topKRegions_{viewer_id}.length : parseInt(sphereCount);
                        numSpheres = Math.min(numSpheres, window.topKRegions_{viewer_id}.length);
                        spheresToShow = window.topKRegions_{viewer_id}.slice(0, numSpheres).map((region, index) => ({{ region, index }}));
                    }}
                    
                    spheresToShow.forEach(({{ region, index }}) => {{
                        const color = colors[index % colors.length];
                        const centerResidues = window.viewer_{viewer_id}.getModel(0).selectedAtoms({{
                            resi: region.center_residue,
                            atom: 'CA'
                        }});
                        
                        if (centerResidues.length > 0) {{
                            const centerAtom = centerResidues[0];
                            const centerCoords = {{ x: centerAtom.x, y: centerAtom.y, z: centerAtom.z }};
                            
                            // Add wireframe sphere
                            window.viewer_{viewer_id}.addSphere({{
                                center: centerCoords,
                                radius: region.radius,
                                color: color,
                                wireframe: true,
                                linewidth: 2.0
                            }});
                            
                            // Add center point
                            window.viewer_{viewer_id}.addSphere({{
                                center: centerCoords,
                                radius: 0.7,
                                color: '#FFD700',
                                wireframe: false
                            }});
                        }}
                    }});
                }}
                
                window.viewer_{viewer_id}.zoomTo();
                window.viewer_{viewer_id}.render();
            }} catch (error) {{
                console.error('Error updating visualization:', error);
            }}
        }}
        
        // Color interpolation helper functions
        function interpolateColor(color1, color2, factor) {{
            const c1 = hexToRgb(color1);
            const c2 = hexToRgb(color2);
            
            const r = Math.round(c1.r + factor * (c2.r - c1.r));
            const g = Math.round(c1.g + factor * (c2.g - c1.g));
            const b = Math.round(c1.b + factor * (c2.b - c1.b));
            
            return rgbToHex(r, g, b);
        }}
        
        function hexToRgb(hex) {{
            const result = /^#?([a-f\d]{{2}})([a-f\d]{{2}})([a-f\d]{{2}})$/i.exec(hex);
            return result ? {{
                r: parseInt(result[1], 16),
                g: parseInt(result[2], 16),
                b: parseInt(result[3], 16)
            }} : null;
        }}
        
        function rgbToHex(r, g, b) {{
            return "#" + ((1 << 24) + (r << 16) + (g << 8) + b).toString(16).slice(1);
        }}
        
        function resetView_{viewer_id}() {{
            if (window.viewer_{viewer_id}) {{
                window.viewer_{viewer_id}.zoomTo();
                window.viewer_{viewer_id}.render();
            }}
        }}
        
        function saveImage_{viewer_id}() {{
            if (window.viewer_{viewer_id}) {{
                window.viewer_{viewer_id}.pngURI(function(uri) {{
                    const link = document.createElement('a');
                    link.href = uri;
                    link.download = '{protein_id}_structure.png';
                    link.click();
                }});
            }}
        }}
        
        // Handle sphere count selection change
        function handleSphereCountChange_{viewer_id}() {{
            const sphereCount = document.getElementById('sphereCount_{viewer_id}').value;
            const customSelectionDiv = document.getElementById('customSphereSelection_{viewer_id}');
            
            if (sphereCount === 'custom') {{
                customSelectionDiv.style.display = 'block';
                generateSphereCheckboxes_{viewer_id}();
            }} else {{
                customSelectionDiv.style.display = 'none';
            }}
            
            updateVisualization_{viewer_id}();
        }}
        
        // Generate sphere checkboxes for custom selection
        function generateSphereCheckboxes_{viewer_id}() {{
            if (!window.topKRegions_{viewer_id} || window.topKRegions_{viewer_id}.length === 0) {{
                return;
            }}
            
            const regions = window.topKRegions_{viewer_id};
            const container = document.getElementById('sphereCheckboxes_{viewer_id}');
            container.innerHTML = '';
            
            regions.forEach((region, index) => {{
                const sphereNum = index + 1;
                const checkboxId = `sphere_{{sphereNum}}_{viewer_id}`;
                const colors = ['#FF6B6B', '#96CEB4', '#4ECDC4', '#45B7D1', '#FFEAA7', '#DDA0DD', '#87CEEB'];
                const sphereColor = colors[index % colors.length];
                
                const checkboxContainer = document.createElement('div');
                checkboxContainer.style.cssText = `
                    display: flex;
                    align-items: center;
                    padding: 5px 10px;
                    border: 1px solid #ddd;
                    border-radius: 4px;
                    background: white;
                    cursor: pointer;
                    user-select: none;
                `;
                checkboxContainer.setAttribute('data-sphere', sphereNum);
                
                const checkbox = document.createElement('input');
                checkbox.type = 'checkbox';
                checkbox.id = checkboxId;
                checkbox.checked = sphereNum <= 5; // Default: show first 5
                checkbox.style.marginRight = '5px';
                
                const colorBox = document.createElement('div');
                colorBox.style.cssText = `
                    width: 16px;
                    height: 16px;
                    background-color: ${{sphereColor}};
                    border: 1px solid #333;
                    border-radius: 2px;
                    margin-right: 5px;
                `;
                
                const label = document.createElement('label');
                label.setAttribute('for', checkboxId);
                label.textContent = `Sphere ${{sphereNum}} (R${{region.center_residue}})`;
                label.style.cursor = 'pointer';
                label.style.fontSize = '14px';
                
                checkboxContainer.appendChild(checkbox);
                checkboxContainer.appendChild(colorBox);
                checkboxContainer.appendChild(label);
                container.appendChild(checkboxContainer);
                
                // Add click handler
                checkboxContainer.addEventListener('click', function(e) {{
                    if (e.target.type !== 'checkbox') {{
                        checkbox.checked = !checkbox.checked;
                    }}
                    
                    if (checkbox.checked) {{
                        checkboxContainer.style.backgroundColor = '#f0f8ff';
                        checkboxContainer.style.borderColor = '#4a90e2';
                    }} else {{
                        checkboxContainer.style.backgroundColor = 'white';
                        checkboxContainer.style.borderColor = '#ddd';
                    }}
                    
                    updateVisualization_{viewer_id}();
                }});
                
                // Initialize visual state
                if (checkbox.checked) {{
                    checkboxContainer.style.backgroundColor = '#f0f8ff';
                    checkboxContainer.style.borderColor = '#4a90e2';
                }}
            }});
        }}
        
        // Get selected sphere indices for custom mode
        function getSelectedSphereIndices_{viewer_id}() {{
            const selected = [];
            const checkboxes = document.querySelectorAll('#sphereCheckboxes_{viewer_id} input[type="checkbox"]:checked');
            checkboxes.forEach(function(checkbox) {{
                // Get sphere number from the data-sphere attribute of the container
                const container = checkbox.closest('[data-sphere]');
                if (container) {{
                    const sphereNum = parseInt(container.getAttribute('data-sphere'));
                    selected.push(sphereNum - 1); // Convert to 0-based index
                }}
            }});
            return selected;
        }}
        
        // Start initialization
        wait3Dmol_{viewer_id}();
    </script>
    """
    
    return html_content

def predict_epitopes(pdb_id: str, pdb_file, chain_id: str, radius: float, k: int, 
                    encoder: str, device_config: str, use_threshold: bool, threshold: float,
                    auto_cleanup: bool, progress: gr.Progress = None) -> Tuple[str, str, str, str, str, str]:
    """
    Main prediction function that handles the epitope prediction workflow
    """
    try:
        # Input validation
        if not pdb_file and not pdb_id:
            return "Error: Please provide either a PDB ID or upload a PDB file", "", "", "", "", ""
        
        if pdb_id and not validate_pdb_id(pdb_id):
            return "Error: PDB ID must be exactly 4 characters (letters and numbers)", "", "", "", "", ""
        
        if not validate_chain_id(chain_id):
            return "Error: Chain ID must be exactly 1 character", "", "", "", "", ""
        
        # Update progress
        if progress:
            progress(0.1, desc="Initializing prediction...")
        
        # Process device configuration
        device_id = -1 if device_config == "CPU Only" else int(device_config.split(" ")[1])
        use_gpu = device_id >= 0
        
        # Load protein structure
        if progress:
            progress(0.2, desc="Loading protein structure...")
        
        antigen_chain = None
        temp_file_path = None
        
        try:
            if pdb_file:
                # Handle uploaded file
                if progress:
                        progress(0.25, desc="Processing uploaded PDB file...")
                
                # Debug: print type and attributes of pdb_file
                print(f"🔍 Debug: pdb_file type = {type(pdb_file)}")
                print(f"🔍 Debug: pdb_file attributes = {dir(pdb_file)}")
                
                # Extract PDB ID from filename if not provided
                if not pdb_id:
                    if hasattr(pdb_file, 'name'):
                        pdb_id = Path(pdb_file.name).stem.split('_')[0][:4]
                    else:
                        pdb_id = "UNKN"  # Default fallback
                
                # Save uploaded file to data/pdb/ directory with proper naming
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"{pdb_id}_{chain_id}_{timestamp}.pdb"
                temp_file_path = PDB_DATA_DIR / filename
                
                # Properly read and write the uploaded file
                try:
                    if hasattr(pdb_file, 'name') and os.path.isfile(pdb_file.name):
                        # pdb_file is a file object with .name attribute
                        print(f"📁 Processing file object: {pdb_file.name}")
                        with open(pdb_file.name, "rb") as src:
                            with open(temp_file_path, "wb") as dst:
                                dst.write(src.read())
                    elif hasattr(pdb_file, 'read'):
                        # pdb_file is a file-like object
                        print(f"📄 Processing file-like object")
                        with open(temp_file_path, "wb") as f:
                            f.write(pdb_file.read())
                    else:
                        # pdb_file is a string (file path)
                        print(f"📍 Processing file path: {pdb_file}")
                        with open(str(pdb_file), "rb") as src:
                            with open(temp_file_path, "wb") as dst:
                                dst.write(src.read())
                    
                    print(f"✅ PDB file saved to: {temp_file_path}")
                    
                except Exception as file_error:
                    print(f"❌ Error processing uploaded file: {file_error}")
                    return f"Error processing uploaded file: {str(file_error)}", "", "", "", "", ""
                
                antigen_chain = AntigenChain.from_pdb(
                    path=str(temp_file_path),
                    chain_id=chain_id,
                    id=pdb_id
                )
            else:
                # Load from PDB ID
                if progress:
                    progress(0.25, desc=f"Downloading PDB structure {pdb_id}...")
                
                antigen_chain = AntigenChain.from_pdb(
                    chain_id=chain_id,
                    id=pdb_id
                )
                
        except Exception as e:
            return f"Error loading protein structure: {str(e)}", "", "", "", "", ""
        
        if antigen_chain is None:
            return "Error: Failed to load protein structure", "", "", "", "", ""
        
        # Run prediction
        if progress:
            progress(0.4, desc="Running epitope prediction...")
        
        try:
            # Use threshold only if checkbox is checked
            final_threshold = threshold if use_threshold else None
            
            predict_results = antigen_chain.predict(
                model_path=DEFAULT_MODEL_PATH,
                device_id=device_id,
                radius=radius,
                k=k,
                threshold=final_threshold,
                verbose=True,
                encoder=encoder,
                use_gpu=use_gpu,
                auto_cleanup=auto_cleanup
            )
        except Exception as e:
            error_msg = f"Error during prediction: {str(e)}"
            print(f"Prediction error: {error_msg}")
            import traceback
            traceback.print_exc()
            return error_msg, "", "", "", "", ""
        
        if progress:
            progress(0.8, desc="Processing results...")
        
        # Process results
        if not predict_results:
            return "Error: No prediction results generated", "", "", "", "", ""
        
        # Extract prediction data
        predicted_epitopes = predict_results.get("predicted_epitopes", [])
        predictions = predict_results.get("predictions", {})
        top_k_centers = predict_results.get("top_k_centers", [])
        top_k_region_residues = predict_results.get("top_k_region_residues", [])
        top_k_regions = predict_results.get("top_k_regions", [])
        
        # Calculate summary statistics
        protein_length = len(antigen_chain.sequence)
        epitope_count = len(predicted_epitopes)
        region_count = len(top_k_regions)
        top_k_region_residues_count = len(top_k_region_residues)
        coverage_rate = (len(top_k_region_residues) / protein_length) * 100 if protein_length > 0 else 0
        
        # Create summary text
        summary_text = f"""
## Prediction Results for {pdb_id}_{chain_id}

### Protein Information
- **PDB ID**: {pdb_id}
- **Chain**: {chain_id}
- **Length**: {protein_length} residues
- **Sequence**: <div style="word-wrap: break-word; word-break: break-all; white-space: pre-wrap; max-width: 100%; font-family: monospace; background: #f5f5f5; padding: 8px; border-radius: 4px; margin: 5px 0; display: inline-block;">{antigen_chain.sequence}</div>

### Prediction Summary
- **Number of Predicted Epitope Residues**: {epitope_count}
- **Top-k Regions**: {region_count}
- **Number of Residues in Predicted Binding Regions**: {top_k_region_residues_count}

### Top-k Region Centers
{', '.join(map(str, top_k_centers))}

### Predicted Epitope Residues
{', '.join(map(str, predicted_epitopes))}

### Binding Region Residues (Top-k Union)
{', '.join(map(str, top_k_region_residues))}
        """
        
        # Create epitope list text with residue names
        epitope_text = f"Predicted Epitope Residues ({len(predicted_epitopes)}):\n"
        epitope_lines = []
        for res in predicted_epitopes:
            # Get residue index from residue number
            if res in antigen_chain.resnum_to_index:
                res_idx = antigen_chain.resnum_to_index[res]
                res_name = antigen_chain.sequence[res_idx]
                epitope_lines.append(f"Residue {res} ({res_name})")
            else:
                epitope_lines.append(f"Residue {res}")
        epitope_text += "\n".join(epitope_lines)
        
        # Create binding region text with residue names
        binding_text = f"Binding Region Residues ({len(top_k_region_residues)}):\n"
        binding_lines = []
        for res in top_k_region_residues:
            # Get residue index from residue number
            if res in antigen_chain.resnum_to_index:
                res_idx = antigen_chain.resnum_to_index[res]
                res_name = antigen_chain.sequence[res_idx]
                binding_lines.append(f"Residue {res} ({res_name})")
            else:
                binding_lines.append(f"Residue {res}")
        binding_text += "\n".join(binding_lines)
        
        # Create downloadable files
        if progress:
            progress(0.9, desc="Preparing download files...")
        
        # JSON file
        json_data = {
            "protein_info": {
                "id": pdb_id,
                "chain_id": chain_id,
                "length": protein_length,
                "sequence": antigen_chain.sequence
            },
            "prediction": {
                "predicted_epitopes": predicted_epitopes,
                "predictions": predictions,
                "top_k_centers": top_k_centers,
                "top_k_region_residues": top_k_region_residues,
                "top_k_regions": [
                    {
                        "center_idx": region.get('center_idx', 0),
                        "graph_pred": region.get('graph_pred', 0),
                        "covered_indices": region.get('covered_indices', [])
                    }
                    for region in top_k_regions
                ],
                "coverage_rate": coverage_rate,
                "mean_region_value": 0 # No longer calculated
            },
            "parameters": {
                "radius": radius,
                "k": k,
                "encoder": encoder,
                "device_config": device_config,
                "use_threshold": use_threshold,
                "threshold": final_threshold,
                "auto_cleanup": auto_cleanup
            }
        }
        
        # Save JSON file
        json_file_path = tempfile.mktemp(suffix=".json")
        with open(json_file_path, "w") as f:
            json.dump(json_data, f, indent=2)
        
        # CSV file  
        csv_data = []
        for i, residue_num in enumerate(antigen_chain.residue_index):
            residue_num = int(residue_num)
            csv_data.append({
                "Residue_Number": residue_num,
                "Residue_Type": antigen_chain.sequence[i],
                "Prediction_Probability": predictions.get(residue_num, 0.0),
                "Is_Predicted_Epitope": 1 if residue_num in predicted_epitopes else 0,
                "Is_In_TopK_Regions": 1 if residue_num in top_k_region_residues else 0
            })
        
        csv_df = pd.DataFrame(csv_data)
        csv_file_path = tempfile.mktemp(suffix=".csv")
        csv_df.to_csv(csv_file_path, index=False)
        
        # Create 3D visualization
        if progress:
            progress(0.95, desc="Creating 3D visualization...")
        
        # Generate PDB string for visualization HTML file
        html_file_path = None
        try:
            pdb_str = generate_pdb_string(antigen_chain)
            html_content = create_pdb_visualization_html(
                pdb_str, predicted_epitopes, predictions, f"{pdb_id}_{chain_id}", top_k_regions
            )
            
            # Save HTML file to data directory for download
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            html_filename = f"{pdb_id}_{chain_id}_visualization_{timestamp}.html"
            html_file_path = PDB_DATA_DIR / html_filename
            
            with open(html_file_path, "w", encoding='utf-8') as f:
                f.write(html_content)
                
            print(f"✅ 3D visualization HTML saved to: {html_file_path}")
                
        except Exception as e:
            html_file_path = None
            print(f"Warning: Could not create 3D visualization: {str(e)}")
        
        # Clean up temporary files if auto_cleanup is enabled
        if auto_cleanup and temp_file_path and os.path.exists(temp_file_path):
            os.remove(temp_file_path)
            print(f"🧹 Cleaned up temporary file: {temp_file_path}")
        elif temp_file_path and os.path.exists(temp_file_path):
            print(f"📁 PDB file retained at: {temp_file_path}")
        
        if progress:
            progress(1.0, desc="Prediction completed!")
        
        # Return all results including HTML file path for download
        return (
            summary_text,
            epitope_text,
            binding_text,
            str(html_file_path) if html_file_path else None,  # HTML file moved to 4th position
            json_file_path,
            csv_file_path
        )
        
    except Exception as e:
        import traceback
        error_msg = f"Error: {str(e)}\n\nTraceback:\n{traceback.format_exc()}"
        return error_msg, "", "", "", "", ""

def generate_pdb_string(antigen_chain) -> str:
    """Generate PDB string for 3D visualization"""
    from esm.utils import residue_constants as RC
    
    pdb_str = "MODEL        1\n"
    atom_num = 1
    
    for res_idx in range(len(antigen_chain.sequence)):
        one_letter = antigen_chain.sequence[res_idx]
        resname = antigen_chain.convert_letter_1to3(one_letter)
        resnum = antigen_chain.residue_index[res_idx]
        
        mask = antigen_chain.atom37_mask[res_idx]
        coords = antigen_chain.atom37_positions[res_idx][mask]
        atoms = [name for name, exists in zip(RC.atom_types, mask) if exists]
        
        for atom_name, coord in zip(atoms, coords):
            x, y, z = coord
            pdb_str += (f"ATOM  {atom_num:5d}  {atom_name:<3s} {resname:>3s} {antigen_chain.chain_id:1s}{resnum:4d}"
                       f"    {x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00\n")
            atom_num += 1
    
    pdb_str += "ENDMDL\n"
    return pdb_str

def create_interface():
    """Create the Gradio interface"""

    with gr.Blocks(css="""
        .container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }
        .header {
            text-align: center;
            margin-bottom: 30px;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border-radius: 10px;
        }
        .header h1 {
            font-size: 2.5em;
            margin-bottom: 10px;
        }
        .form-row {
            display: flex;
            gap: 20px;
            align-items: end;
        }
        .form-row > * {
            flex: 1;
        }
        .section {
            margin: 20px 0;
            padding: 15px;
            background: #f8f9fa;
            border-radius: 8px;
            border-left: 4px solid #007bff;
        }
        .section h2 {
            color: #333;
            margin-bottom: 15px;
        }
        .results-section {
            margin-top: 30px;
            padding: 20px;
            background: #f0f8ff;
            border-radius: 8px;
            border: 1px solid #e0e8f0;
        }
        .download-section {
            margin-top: 20px;
            padding: 15px;
            background: #f9f9f9;
            border-radius: 8px;
        }
        .download-section h3 {
            color: #333;
            margin-bottom: 10px;
        }
        """) as interface:
        
        # Header
        gr.HTML("""
        <div class="header">
            <h1>🧬 B-cell Epitope Prediction Server</h1>
            <p>Predict epitopes using the RoBep model</p>
        </div>
        """)

        with gr.Row():
            with gr.Column(scale=1):
                gr.HTML("<div class='section'><h2>📋 Input Protein Structure</h2></div>")

                input_method = gr.Radio(
                    choices=["PDB ID", "Upload PDB File"],
                    value="PDB ID",
                    label="Input Method"
                )

                pdb_id = gr.Textbox(
                    label="PDB ID", 
                    placeholder="e.g., 5I9Q", 
                    max_lines=1, 
                    visible=True
                )
                pdb_file = gr.File(
                    label="Upload PDB File", 
                    file_types=[".pdb", ".ent"], 
                    visible=False
                )
                chain_id = gr.Textbox(
                    label="Chain ID", 
                    value="A", 
                    max_lines=1
                )

                with gr.Accordion("🔧 Advanced Parameters", open=False):
                    radius = gr.Slider(
                        label="Radius (Å)", 
                        minimum=1.0, 
                        maximum=50.0, 
                        step=0.1, 
                        value=18.0
                    )
                    k = gr.Slider(
                        label="Top-k Regions", 
                        minimum=1, 
                        maximum=20, 
                        step=1, 
                        value=7
                    )
                    encoder = gr.Dropdown(
                        label="Encoder", 
                        choices=["esmc", "esm2"], 
                        value="esmc"
                    )
                    device_config = gr.Dropdown(
                        label="Device Configuration", 
                        choices=["CPU Only", "GPU 0", "GPU 1", "GPU 2", "GPU 3"], 
                        value="CPU Only"
                    )
                    use_threshold = gr.Checkbox(
                        label="Use Custom Threshold", 
                        value=False
                    )
                    threshold = gr.Number(
                        label="Threshold Value", 
                        value=0.366, 
                        visible=False
                    )
                    auto_cleanup = gr.Checkbox(
                        label="Auto-cleanup Generated Data", 
                        value=True
                    )

                predict_btn = gr.Button("🧮 Predict Epitopes", variant="primary", size="lg")

            with gr.Column(scale=2):
                gr.HTML("<div class='section'><h2>📊 Prediction Results</h2></div>")
                
                # 3D Visualization download (moved to top)
                gr.HTML("<div style='margin: 15px 0; padding: 10px; background: #f0f8ff; border-left: 4px solid #4a90e2; border-radius: 5px;'><h3 style='margin: 0 0 8px 0; color: #333;'>🧬 3D Visualization</h3><p style='margin: 0; color: #666;'>You can download the HTML to visualize the prediction results and the spheres used.</p></div>")
                html_download = gr.File(
                    label="Download Interactive 3D Visualization HTML",
                    visible=True
                )
                results_text = gr.Markdown(label="Prediction Summary", visible=True)

                with gr.Row():
                    epitope_list = gr.Textbox(
                        label="Predicted Epitope Residues", 
                        max_lines=10, 
                        interactive=False
                    )
                    binding_regions = gr.Textbox(
                        label="Binding Region Residues", 
                        max_lines=10, 
                        interactive=False
                    )

                gr.HTML("<div class='download-section'><h3>📥 Download Data Results</h3></div>")
                with gr.Row():
                    json_download = gr.File(
                        label="JSON Results",
                        visible=True
                    )
                    csv_download = gr.File(
                        label="CSV Results",
                        visible=True
                    )

        def toggle_input_method(method):
            return (gr.update(visible=method == "PDB ID"),
                    gr.update(visible=method == "Upload PDB File"))

        def toggle_threshold(use_threshold):
            return gr.update(visible=use_threshold)

        input_method.change(toggle_input_method, inputs=[input_method], outputs=[pdb_id, pdb_file])
        use_threshold.change(toggle_threshold, inputs=[use_threshold], outputs=[threshold])

        predict_btn.click(
            predict_epitopes,
            inputs=[
                pdb_id, pdb_file, chain_id, radius, k, encoder, 
                device_config, use_threshold, threshold, auto_cleanup
            ],
            outputs=[
                results_text, epitope_list, binding_regions, 
                html_download, json_download, csv_download
            ],
            show_progress=True
        )

        gr.HTML("""
        <div style="text-align: center; margin-top: 30px; padding: 20px; background: #f0f0f0; border-radius: 10px;">
            <p>© 2024 B-cell Epitope Prediction Server | Powered by RoBep model</p>
            <p><strong>Features:</strong> PDB ID/File support • 3D visualization • Multiple export formats</p>
        </div>
        """)

    return interface


if __name__ == "__main__":
    # Create and launch the interface
    try:
        interface = create_interface()
        
        # Check if running on Hugging Face Spaces
        is_spaces = os.getenv("SPACE_ID") is not None
        
        interface.launch(
            server_name="0.0.0.0",
            server_port=7860,
            share=is_spaces,  # Use share=True on Spaces, False locally
            show_error=True,
            max_threads=4 if is_spaces else 8
        )
    except Exception as e:
        print(f"Error launching application: {e}")
        print("Please ensure all dependencies are installed correctly.")
        import traceback
        traceback.print_exc()
