from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.requests import Request
import uvicorn
import os
import json
import tempfile
import traceback
from pathlib import Path
from typing import Optional, Dict, Any
import asyncio
from datetime import datetime

# Import configuration and antigen module
import config
import sys
sys.path.append(str(config.BASE_DIR))  # Add project root to path
from bce.antigen.antigen import AntigenChain

app = FastAPI(title="BCE Prediction Server", description="B-cell Epitope Prediction Web Server")

# Setup static files and templates
app.mount("/static", StaticFiles(directory=config.STATIC_DIR), name="static")
templates = Jinja2Templates(directory=config.TEMPLATES_DIR)

# Global storage for prediction results (in production, use Redis or database)
prediction_results = {}

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Home page with input form"""
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict")
async def predict_epitopes(
    background_tasks: BackgroundTasks,
    pdb_id: Optional[str] = Form(None),
    chain_id: str = Form("A"),
    pdb_file: Optional[UploadFile] = File(None),
    model_path: Optional[str] = Form(None),
    radius: float = Form(19.0),
    k: int = Form(7),
    threshold: Optional[float] = Form(None),
    encoder: str = Form("esmc"),
    device_id: int = Form(-1),
    auto_cleanup: str = Form("false")
):
    """Start epitope prediction task"""
    
    # Generate unique task ID
    task_id = f"task_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
    
    # Initialize task status
    prediction_results[task_id] = {
        "status": "processing",
        "progress": 0,
        "message": "Initializing prediction...",
        "result": None,
        "error": None
    }
    
    # Start background task
    background_tasks.add_task(
        run_prediction,
        task_id,
        pdb_id,
        chain_id,
        pdb_file,
        model_path,
        radius,
        k,
        threshold,
        encoder,
        device_id,
        auto_cleanup
    )
    
    return {"task_id": task_id, "status": "started"}

async def run_prediction(
    task_id: str,
    pdb_id: Optional[str],
    chain_id: str,
    pdb_file: Optional[UploadFile],
    model_path: Optional[str],
    radius: float,
    k: int,
    threshold: Optional[float],
    encoder: str,
    device_id: int,
    auto_cleanup: str
):
    """Run the actual prediction in background"""
    try:
        # Update status
        prediction_results[task_id]["message"] = "Loading protein structure..."
        prediction_results[task_id]["progress"] = 10
        
        # Load protein structure
        antigen_chain = None
        temp_file_path = None
        
        if pdb_file:
            # Handle uploaded file
            prediction_results[task_id]["message"] = "Processing uploaded PDB file..."
            prediction_results[task_id]["progress"] = 15
            
            temp_file_path = f"{config.TEMP_DIR}/{task_id}_{pdb_file.filename}"
            with open(temp_file_path, "wb") as f:
                content = await pdb_file.read()
                f.write(content)
            
            # Extract PDB ID from filename if not provided
            if not pdb_id:
                pdb_id = Path(pdb_file.filename).stem.split('_')[0]
            
            prediction_results[task_id]["message"] = "Loading structure from file..."
            prediction_results[task_id]["progress"] = 20
            
            antigen_chain = AntigenChain.from_pdb(
                path=temp_file_path,
                chain_id=chain_id,
                id=pdb_id
            )
        elif pdb_id:
            # Load from PDB ID
            prediction_results[task_id]["message"] = f"Downloading PDB structure {pdb_id}..."
            prediction_results[task_id]["progress"] = 15
            
            antigen_chain = AntigenChain.from_pdb(
                chain_id=chain_id,
                id=pdb_id
            )
        else:
            raise ValueError("Either PDB ID or PDB file must be provided")
        
        if antigen_chain is None:
            raise ValueError("Failed to load protein structure")
        
        prediction_results[task_id]["message"] = "Preparing protein embeddings..."
        prediction_results[task_id]["progress"] = 25
        
        # Small delay for progress update
        await asyncio.sleep(0.1)
        
        prediction_results[task_id]["message"] = "Running epitope prediction..."
        prediction_results[task_id]["progress"] = 30
        
        # Determine whether to use GPU and set proper device_id
        use_gpu = device_id >= 0  # Use GPU if device_id is non-negative
        actual_device_id = device_id if use_gpu else 0  # Use 0 as default GPU ID even if not using GPU
        
        # Debug information
        print(f"[DEBUG] Original device_id: {device_id} (type: {type(device_id)})")
        print(f"[DEBUG] use_gpu: {use_gpu}")
        print(f"[DEBUG] actual_device_id: {actual_device_id} (type: {type(actual_device_id)})")
        
        # Run prediction with progress updates
        prediction_results[task_id]["message"] = "Loading AI model..."
        prediction_results[task_id]["progress"] = 35
        
        # Small delay for progress update
        await asyncio.sleep(0.1)
        
        prediction_results[task_id]["message"] = "Analyzing surface regions..."
        prediction_results[task_id]["progress"] = 45
        
        # Small delay for progress update
        await asyncio.sleep(0.1)
        
        predict_results = antigen_chain.predict(
            model_path=model_path,
            device_id=actual_device_id,
            radius=radius,
            k=k,
            threshold=threshold,
            verbose=True,  # Enable verbose for debugging
            encoder=encoder,
            use_gpu=use_gpu,
            auto_cleanup=auto_cleanup.lower() == "true"  # Convert string to boolean
        )
        
        prediction_results[task_id]["message"] = "Processing prediction results..."
        prediction_results[task_id]["progress"] = 85
        
        # Small delay for progress update
        await asyncio.sleep(0.1)
        
        prediction_results[task_id]["message"] = "Generating visualization data..."
        prediction_results[task_id]["progress"] = 90
        
        # Prepare visualization data
        viz_data = prepare_visualization_data(antigen_chain, predict_results)
        
        # Final updates
        prediction_results[task_id]["message"] = "Finalizing results..."
        prediction_results[task_id]["progress"] = 95
        
        # Small delay for final progress update
        await asyncio.sleep(0.1)
        
        # Complete
        prediction_results[task_id]["status"] = "completed"
        prediction_results[task_id]["progress"] = 100
        prediction_results[task_id]["message"] = "Prediction completed successfully"
        prediction_results[task_id]["result"] = {
            "prediction": predict_results,
            "visualization": viz_data,
            "protein_info": {
                "id": antigen_chain.id,
                "chain_id": antigen_chain.chain_id,
                "sequence": antigen_chain.sequence,
                "length": len(antigen_chain.sequence)
            }
        }
        
        # Clean up temporary file
        if temp_file_path and os.path.exists(temp_file_path):
            os.remove(temp_file_path)
            
    except Exception as e:
        error_msg = str(e)
        error_trace = traceback.format_exc()
        
        # Log detailed error information
        print(f"Error in prediction task {task_id}: {error_msg}")
        print(f"Full traceback: {error_trace}")
        
        prediction_results[task_id]["status"] = "error"
        prediction_results[task_id]["error"] = error_msg
        prediction_results[task_id]["message"] = f"Error: {error_msg}"
        
        # Clean up temporary file in case of error
        if temp_file_path and os.path.exists(temp_file_path):
            os.remove(temp_file_path)

def prepare_visualization_data(antigen_chain: AntigenChain, predict_results: Dict[str, Any]) -> Dict[str, Any]:
    """Prepare data for 3D visualization"""
    from esm.utils import residue_constants as RC
    
    # Generate PDB string for 3Dmol.js
    pdb_str = "MODEL        1\n"
    atom_num = 1
    
    for res_idx in range(len(antigen_chain.sequence)):
        one_letter = antigen_chain.sequence[res_idx]
        resname = AntigenChain.convert_letter_1to3(one_letter)
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
    
    # Get top_k_regions and add radius information
    top_k_regions = predict_results.get("top_k_regions", [])
    for region in top_k_regions:
        if "radius" not in region:
            region["radius"] = 19.0  # Default radius used in prediction
    
    return {
        "pdb_data": pdb_str,
        "predicted_epitopes": predict_results.get("predicted_epitopes", []),
        "predictions": predict_results.get("predictions", {}),
        "top_k_centers": predict_results.get("top_k_centers", []),
        "top_k_regions": top_k_regions
    }

@app.get("/status/{task_id}")
async def get_prediction_status(task_id: str):
    """Get prediction task status"""
    if task_id not in prediction_results:
        raise HTTPException(status_code=404, detail="Task not found")
    
    return prediction_results[task_id]

@app.get("/result/{task_id}")
async def get_prediction_result(task_id: str):
    """Get prediction result"""
    if task_id not in prediction_results:
        raise HTTPException(status_code=404, detail="Task not found")
    
    result = prediction_results[task_id]
    if result["status"] != "completed":
        raise HTTPException(status_code=400, detail="Task not completed")
    
    return result["result"]

@app.delete("/task/{task_id}")
async def delete_task(task_id: str):
    """Delete a task and its results"""
    if task_id in prediction_results:
        del prediction_results[task_id]
        return {"message": "Task deleted successfully"}
    else:
        raise HTTPException(status_code=404, detail="Task not found")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "message": "BCE Prediction Server is running"}

if __name__ == "__main__":
    # Create necessary directories
    config.STATIC_DIR.mkdir(exist_ok=True)
    config.TEMPLATES_DIR.mkdir(exist_ok=True)
    
    # Validate configuration
    errors = config.validate_config()
    if errors:
        print("Configuration warnings:")
        for error in errors:
            print(f"  - {error}")
    
    uvicorn.run(app, host=config.SERVER_HOST, port=config.SERVER_PORT, reload=config.DEBUG_MODE)
