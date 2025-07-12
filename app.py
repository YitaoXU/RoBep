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

# 动态安装PyTorch Geometric依赖包
def install_torch_geometric_deps():
    """在运行时安装PyTorch Geometric依赖包，避免Hugging Face Spaces构建时的编译问题"""
    import subprocess
    import sys
    
    # 检查是否已经安装torch-scatter
    try:
        import torch_scatter
        print("✅ torch-scatter already installed")
        return True
    except ImportError:
        print("🔄 Installing torch-scatter and related packages...")
        
        # 获取PyTorch版本和CUDA信息
        torch_version = torch.__version__
        torch_version_str = '+'.join(torch_version.split('+')[:1])  # 移除CUDA信息
        
        # 使用PyTorch Geometric官方推荐的安装方式
        try:
            # 对于CPU版本，使用官方CPU wheel
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
                # 尝试简化安装方式
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

# 尝试安装PyTorch Geometric依赖包
deps_installed = install_torch_geometric_deps()

if not deps_installed:
    print("⚠️ Warning: PyTorch Geometric dependencies not installed. Some features may not work.")
    print("The application will try to continue with limited functionality.")

# Set up paths and imports for different deployment environments
import sys
BASE_DIR = Path(__file__).parent

# Smart import handling for different environments
def setup_imports():
    """智能导入设置，适配不同的部署环境"""
    global AntigenChain, PROJECT_BASE_DIR
    
    # 方案1: 尝试从src目录导入（本地开发）
    if (BASE_DIR / "src").exists():
        sys.path.insert(0, str(BASE_DIR))
        try:
            from src.bce.antigen.antigen import AntigenChain
            from src.bce.utils.constants import BASE_DIR as PROJECT_BASE_DIR
            print("✅ Successfully imported from src/ directory")
            return True
        except ImportError as e:
            print(f"❌ Failed to import from src/: {e}")
    
    # 方案2: 尝试添加src到路径并直接导入（Hugging Face Spaces）
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
    
    # 方案3: 尝试直接导入（如果已安装包）
    try:
        from bce.antigen.antigen import AntigenChain
        from bce.utils.constants import BASE_DIR as PROJECT_BASE_DIR
        print("✅ Successfully imported from installed package")
        return True
    except ImportError as e:
        print(f"❌ Failed to import from installed package: {e}")
    
    # 如果所有方案都失败，使用默认设置
    print("⚠️ All import methods failed, using fallback settings")
    PROJECT_BASE_DIR = BASE_DIR
    return False

# 执行导入设置
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
                                 predictions: dict, protein_id: str) -> str:
    """Create HTML with 3Dmol.js visualization"""
    
    # Create color mapping for residues
    epitope_residues = set(predicted_epitopes)
    
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <script src="https://3Dmol.org/build/3Dmol-min.js"></script>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            #viewer {{ width: 100%; height: 500px; position: relative; }}
            .controls {{ margin: 10px 0; }}
            .control-group {{ margin: 10px 0; }}
            label {{ display: inline-block; width: 150px; font-weight: bold; }}
            select, button {{ margin: 5px; padding: 5px; }}
        </style>
    </head>
    <body>
        <h2>3D Structure Visualization - {protein_id}</h2>
        <div class="controls">
            <div class="control-group">
                <label>Display Mode:</label>
                <select id="vizMode" onchange="updateVisualization()">
                    <option value="prediction">Predicted Epitopes</option>
                    <option value="probability">Probability Gradient</option>
                </select>
            </div>
            <div class="control-group">
                <label>Representation:</label>
                <select id="vizStyle" onchange="updateVisualization()">
                    <option value="cartoon">Cartoon</option>
                    <option value="surface">Surface</option>
                    <option value="stick">Stick</option>
                    <option value="sphere">Sphere</option>
                </select>
            </div>
            <div class="control-group">
                <button onclick="resetView()">Reset View</button>
                <button onclick="saveImage()">Save Image</button>
            </div>
        </div>
        <div id="viewer"></div>
        
        <script>
            let viewer;
            let pdbData = `{pdb_data}`;
            let predictedEpitopes = {json.dumps(predicted_epitopes)};
            let predictions = {json.dumps(predictions)};
            
            function initializeViewer() {{
                viewer = $3Dmol.createViewer('viewer', {{
                    defaultcolors: $3Dmol.rasmolElementColors
                }});
                
                viewer.addModel(pdbData, 'pdb');
                updateVisualization();
            }}
            
            function updateVisualization() {{
                viewer.removeAllModels();
                viewer.addModel(pdbData, 'pdb');
                
                const mode = document.getElementById('vizMode').value;
                const style = document.getElementById('vizStyle').value;
                
                // Base style
                const baseStyle = getBaseStyle(style);
                viewer.setStyle({{}}, baseStyle);
                
                if (mode === 'prediction') {{
                    // Highlight predicted epitopes
                    const epitopeStyle = Object.assign({{}}, baseStyle);
                    epitopeStyle[Object.keys(baseStyle)[0]].color = '#9C6ADE';
                    
                    predictedEpitopes.forEach(resnum => {{
                        viewer.setStyle({{resi: resnum}}, epitopeStyle);
                    }});
                }} else if (mode === 'probability') {{
                    // Color by probability
                    for (const [resnum, prob] of Object.entries(predictions)) {{
                        const color = getProbabilityColor(prob);
                        const probStyle = Object.assign({{}}, baseStyle);
                        probStyle[Object.keys(baseStyle)[0]].color = color;
                        viewer.setStyle({{resi: parseInt(resnum)}}, probStyle);
                    }}
                }}
                
                viewer.zoomTo();
                viewer.render();
            }}
            
            function getBaseStyle(style) {{
                const styles = {{
                    'cartoon': {{cartoon: {{color: '#e6e6f7'}}}},
                    'surface': {{surface: {{color: '#e6e6f7', opacity: 0.7}}}},
                    'stick': {{stick: {{color: '#e6e6f7'}}}},
                    'sphere': {{sphere: {{color: '#e6e6f7'}}}}
                }};
                return styles[style] || styles['cartoon'];
            }}
            
            function getProbabilityColor(prob) {{
                // Color gradient from blue (low) to red (high)
                const r = Math.floor(prob * 255);
                const b = Math.floor((1 - prob) * 255);
                return `rgb(${{r}}, 0, ${{b}})`;
            }}
            
            function resetView() {{
                viewer.zoomTo();
                viewer.render();
            }}
            
            function saveImage() {{
                const canvas = viewer.pngURI();
                const link = document.createElement('a');
                link.download = '{protein_id}_structure.png';
                link.href = canvas;
                link.click();
            }}
            
            // Initialize when page loads
            document.addEventListener('DOMContentLoaded', initializeViewer);
        </script>
    </body>
    </html>
    """
    
    return html_content

def predict_epitopes(pdb_id: str, pdb_file, chain_id: str, radius: float, k: int, 
                    encoder: str, device_config: str, threshold: Optional[float],
                    auto_cleanup: bool, progress: gr.Progress) -> Tuple[str, str, str, str, str, str]:
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
        progress(0.1, desc="Initializing prediction...")
        
        # Process device configuration
        device_id = -1 if device_config == "CPU Only" else int(device_config.split(" ")[1])
        use_gpu = device_id >= 0
        
        # Load protein structure
        progress(0.2, desc="Loading protein structure...")
        
        antigen_chain = None
        temp_file_path = None
        
        try:
            if pdb_file:
                # Handle uploaded file
                progress(0.25, desc="Processing uploaded PDB file...")
                
                # Save uploaded file to temporary location
                temp_file_path = tempfile.mktemp(suffix=".pdb")
                with open(temp_file_path, "wb") as f:
                    f.write(pdb_file)
                
                # Extract PDB ID from filename if not provided
                if not pdb_id:
                    pdb_id = Path(pdb_file.name).stem.split('_')[0][:4]
                
                antigen_chain = AntigenChain.from_pdb(
                    path=temp_file_path,
                    chain_id=chain_id,
                    id=pdb_id
                )
            else:
                # Load from PDB ID
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
        progress(0.4, desc="Running epitope prediction...")
        
        try:
            predict_results = antigen_chain.predict(
                model_path=DEFAULT_MODEL_PATH,
                device_id=device_id,
                radius=radius,
                k=k,
                threshold=threshold,
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
        
        # Calculate statistics
        protein_length = len(antigen_chain.sequence)
        epitope_count = len(predicted_epitopes)
        region_count = len(top_k_regions)
        coverage_rate = (len(top_k_region_residues) / protein_length) * 100 if protein_length > 0 else 0
        
        # Calculate mean region prediction value
        region_values = [region.get('graph_pred', 0) for region in top_k_regions]
        mean_region = np.mean(region_values) if region_values else 0
        
        # Calculate epitope rate (antigenicity)
        epitope_rate = (epitope_count / protein_length) * 100 if protein_length > 0 else 0
        
        # Create summary text
        summary_text = f"""
## Prediction Results for {pdb_id}_{chain_id}

### Protein Information
- **PDB ID**: {pdb_id}
- **Chain**: {chain_id}
- **Length**: {protein_length} residues
- **Sequence**: {antigen_chain.sequence}

### Prediction Summary
- **Predicted Epitopes**: {epitope_count}
- **Top-k Regions**: {region_count}
- **Coverage Rate**: {coverage_rate:.1f}%
- **Mean Region Value**: {mean_region:.3f}
- **Epitope Rate**: {epitope_rate:.1f}%

### Top-k Region Centers
{', '.join(map(str, top_k_centers))}

### Predicted Epitope Residues
{', '.join(map(str, predicted_epitopes))}

### Binding Region Residues (Top-k Union)
{', '.join(map(str, top_k_region_residues))}
        """
        
        # Create epitope list text
        epitope_text = f"Predicted Epitope Residues ({len(predicted_epitopes)}):\n"
        epitope_text += "\n".join([f"Residue {res}" for res in predicted_epitopes])
        
        # Create binding region text
        binding_text = f"Binding Region Residues ({len(top_k_region_residues)}):\n"
        binding_text += "\n".join([f"Residue {res}" for res in top_k_region_residues])
        
        # Create downloadable files
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
                "epitope_rate": epitope_rate,
                "coverage_rate": coverage_rate,
                "mean_region_value": mean_region
            },
            "parameters": {
                "radius": radius,
                "k": k,
                "encoder": encoder,
                "device_config": device_config,
                "threshold": threshold,
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
        progress(0.95, desc="Creating 3D visualization...")
        
        # Generate PDB string for visualization
        try:
            pdb_str = generate_pdb_string(antigen_chain)
            html_content = create_pdb_visualization_html(
                pdb_str, predicted_epitopes, predictions, f"{pdb_id}_{chain_id}"
            )
            
            # Save HTML file
            html_file_path = tempfile.mktemp(suffix=".html")
            with open(html_file_path, "w") as f:
                f.write(html_content)
                
        except Exception as e:
            html_file_path = None
            print(f"Warning: Could not create 3D visualization: {str(e)}")
        
        # Clean up temporary files
        if temp_file_path and os.path.exists(temp_file_path):
            os.remove(temp_file_path)
        
        progress(1.0, desc="Prediction completed!")
        
        return (
            summary_text,
            epitope_text,
            binding_text,
            json_file_path,
            csv_file_path,
            html_file_path
        )
        
    except Exception as e:
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
    
    with gr.Blocks(
        title="B-cell Epitope Prediction Server",
        theme=gr.themes.Soft(),
        css="""
        .gradio-container {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
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
        .section {
            margin: 20px 0;
            padding: 20px;
            border-radius: 10px;
            background: #f8f9fa;
        }
        """
    ) as interface:
        
        # Header
        gr.HTML("""
        <div class="header">
            <h1>🧬 B-cell Epitope Prediction Server</h1>
            <p>Predict epitopes using the ReCEP model</p>
        </div>
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.HTML("<div class='section'><h2>📋 Input Protein Structure</h2></div>")
                
                # Input method selection
                input_method = gr.Radio(
                    choices=["PDB ID", "Upload PDB File"],
                    value="PDB ID",
                    label="Input Method"
                )
                
                # PDB ID input
                pdb_id = gr.Textbox(
                    label="PDB ID",
                    placeholder="e.g., 5I9Q",
                    max_lines=1,
                    visible=True
                )
                
                # File upload
                pdb_file = gr.File(
                    label="Upload PDB File",
                    file_types=[".pdb", ".ent"],
                    visible=False
                )
                
                # Chain ID
                chain_id = gr.Textbox(
                    label="Chain ID",
                    value="A",
                    max_lines=1
                )
                
                # Advanced parameters
                with gr.Accordion("🔧 Advanced Parameters", open=False):
                    radius = gr.Slider(
                        label="Radius (Å)",
                        minimum=1.0,
                        maximum=50.0,
                        step=0.1,
                        value=19.0
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
                    
                    threshold = gr.Number(
                        label="Threshold (optional)",
                        minimum=0.0,
                        maximum=1.0,
                        step=0.01,
                        value=None,
                        precision=2
                    )
                    
                    auto_cleanup = gr.Checkbox(
                        label="Auto-cleanup Generated Data",
                        value=True,
                        info="Automatically delete generated files after prediction to save disk space"
                    )
                
                # Predict button
                predict_btn = gr.Button("🧮 Predict Epitopes", variant="primary", size="lg")
                
            with gr.Column(scale=2):
                gr.HTML("<div class='section'><h2>📊 Results</h2></div>")
                
                # Results display
                results_text = gr.Markdown(label="Prediction Summary")
                
                with gr.Row():
                    with gr.Column():
                        epitope_list = gr.Textbox(
                            label="Predicted Epitope Residues",
                            max_lines=10,
                            interactive=False
                        )
                    
                    with gr.Column():
                        binding_regions = gr.Textbox(
                            label="Binding Region Residues",
                            max_lines=10,
                            interactive=False
                        )
                
                # Download section
                with gr.Row():
                    json_download = gr.File(label="📥 Download JSON Results")
                    csv_download = gr.File(label="📥 Download CSV Results")
                
                # 3D Visualization
                with gr.Accordion("🎨 3D Structure Visualization", open=False):
                    gr.HTML("""
                    <p><strong>Note:</strong> The 3D visualization will be available as a downloadable HTML file 
                    that you can open in your browser for interactive viewing.</p>
                    """)
                    html_download = gr.File(label="📥 Download 3D Visualization")
        
        # Event handlers
        def toggle_input_method(method):
            if method == "PDB ID":
                return gr.update(visible=True), gr.update(visible=False)
            else:
                return gr.update(visible=False), gr.update(visible=True)
        
        input_method.change(
            toggle_input_method,
            inputs=[input_method],
            outputs=[pdb_id, pdb_file]
        )
        
        # Prediction function
        predict_btn.click(
            predict_epitopes,
            inputs=[
                pdb_id, pdb_file, chain_id, radius, k, encoder, 
                device_config, threshold, auto_cleanup
            ],
            outputs=[
                results_text, epitope_list, binding_regions, 
                json_download, csv_download, html_download
            ],
            show_progress=True  # 启用进度显示
        )
        
        # Footer
        gr.HTML("""
        <div style="text-align: center; margin-top: 30px; padding: 20px; background: #f0f0f0; border-radius: 10px;">
            <p>© 2024 B-cell Epitope Prediction Server | Powered by ReCEP model</p>
            <p>🚀 Deployed on Hugging Face Spaces</p>
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
            share=not is_spaces,  # Don't create public links on Spaces
            show_error=True,
            enable_queue=True,
            max_threads=4 if is_spaces else 8
        )
    except Exception as e:
        print(f"Error launching application: {e}")
        print("Please ensure all dependencies are installed correctly.")
        import traceback
        traceback.print_exc()
