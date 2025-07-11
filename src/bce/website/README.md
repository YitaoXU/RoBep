# BCE Prediction Web Server

A web-based interface for B-cell epitope prediction using the ReCEP model.

## Features

- 🧬 **Protein Structure Input**: Support for PDB ID lookup or file upload
- 🤖 **AI-Powered Prediction**: Uses the ReCEP model for epitope prediction
- 🎨 **Interactive 3D Visualization**: Real-time molecular structure viewer using 3Dmol.js
- 📊 **Comprehensive Results**: Detailed prediction statistics and epitope analysis
- 💾 **Export Options**: Download results in JSON or CSV format
- ⚡ **Asynchronous Processing**: Background processing with real-time progress updates

## Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-compatible GPU (recommended for faster predictions)
- Sufficient disk space for protein structure files and model weights

### Setup

1. **Navigate to the website directory:**
   ```bash
   cd src/bce/website
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Ensure your ReCEP model is available:**
   - The default model path is configured in the main application
   - You can specify a custom model path through the web interface

## Usage

### Starting the Server

#### Method 1: Using the launch script (Recommended)
```bash
python run_server.py
```

#### Method 2: Direct uvicorn command
```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

#### Command Line Options
```bash
python run_server.py --help

Options:
  --host TEXT          Host to bind to (default: 0.0.0.0)
  --port INTEGER       Port to bind to (default: 8000)
  --reload            Enable auto-reload for development
  --workers INTEGER    Number of worker processes
  --log-level TEXT     Log level (debug, info, warning, error, critical)
```

### Accessing the Web Interface

Once the server is running, open your web browser and navigate to:
- **Main Interface**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health

## Web Interface Guide

### 1. Input Protein Structure

Choose one of two input methods:

#### Option A: PDB ID
- Enter a 4-character PDB ID (e.g., "1FBI")
- Specify the chain ID (e.g., "A")

#### Option B: File Upload
- Upload a PDB structure file (.pdb or .ent format)
- Optionally specify a custom PDB ID
- Specify the chain ID

### 2. Configure Prediction Parameters

#### Basic Parameters:
- **Chain ID**: Target protein chain (default: A)

#### Advanced Parameters (Optional):
- **Radius**: Spherical region radius in Ångstroms (default: 19.0)
- **Top-k Regions**: Number of top regions to analyze (default: 7)
- **Encoder**: Protein encoder type (ESM-C or ESM-2)
- **GPU Device ID**: CUDA device ID for GPU acceleration
- **Threshold**: Custom prediction threshold (leave empty for auto)
- **Model Path**: Custom ReCEP model path (leave empty for default)

### 3. Monitor Prediction Progress

- Real-time progress bar with status updates
- Typical prediction time: 2-5 minutes depending on protein size and hardware

### 4. Analyze Results

#### Protein Information
- PDB ID, chain, and sequence length

#### Prediction Summary
- Number of predicted epitope residues
- Number of analyzed regions
- Coverage rate and mean probability

#### 3D Visualization
- **Display Modes**:
  - Predicted Epitopes: Highlight predicted epitope residues
  - Probability Gradient: Color residues by prediction confidence
  - Top-k Regions: Show all analyzed regions in different colors

- **Representation Styles**:
  - Cartoon: Protein secondary structure
  - Surface: Molecular surface
  - Stick: Atomic detail
  - Sphere: Space-filling model

- **Interactive Controls**:
  - Mouse: Rotate, zoom, and pan
  - Reset view and save image options

#### Epitope Residue List
- Numbered list of all predicted epitope residues
- Sorted by residue number

### 5. Export Results

- **JSON Format**: Complete prediction data with metadata
- **CSV Format**: Residue-level predictions for analysis

## API Endpoints

### Core Endpoints

- `POST /predict`: Submit prediction job
- `GET /status/{task_id}`: Check prediction status
- `GET /result/{task_id}`: Retrieve prediction results
- `DELETE /task/{task_id}`: Delete task and results

### Utility Endpoints

- `GET /health`: Server health check
- `GET /`: Main web interface
- `GET /docs`: Interactive API documentation

## Architecture

### Backend (FastAPI)
- **Asynchronous Processing**: Background tasks for long-running predictions
- **File Handling**: Support for PDB file uploads with validation
- **Error Handling**: Comprehensive error messages and logging
- **REST API**: Clean API design for integration

### Frontend (HTML/CSS/JavaScript)
- **Responsive Design**: Works on desktop and mobile devices
- **Real-time Updates**: AJAX-based progress monitoring
- **3D Visualization**: 3Dmol.js integration for molecular graphics
- **User-Friendly Interface**: Intuitive forms and controls

### Data Flow
1. User submits protein structure and parameters
2. Server validates input and creates background task
3. ReCEP model processes protein structure
4. Results are generated and cached
5. Frontend displays results with 3D visualization

## Configuration

### Environment Variables

You can configure the application using environment variables:

```bash
export BCE_MODEL_PATH="/path/to/your/model"
export BCE_DATA_DIR="/path/to/data"
export BCE_DEVICE_ID="0"
```

### Model Configuration

The application uses the ReCEP model for epitope prediction. Ensure:
- Model weights are accessible at the specified path
- CUDA is available for GPU acceleration (recommended)
- Sufficient memory for large protein structures

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure all dependencies are installed
2. **CUDA Issues**: Check GPU availability and CUDA installation
3. **Memory Errors**: Reduce batch size or use CPU-only mode
4. **File Upload Errors**: Check file format and size limits
5. **Port Conflicts**: Use a different port with `--port` option

### Logging

The application provides detailed logging. Check the console output for:
- Server startup messages
- Request processing logs
- Error details and stack traces

### Performance Tips

- Use GPU acceleration when available
- Close unused browser tabs to free memory
- Use smaller protein structures for testing
- Monitor system resources during prediction

## Development

### Project Structure
```
src/bce/website/
├── main.py              # FastAPI application
├── run_server.py        # Launch script
├── requirements.txt     # Dependencies
├── README.md           # Documentation
├── templates/          # HTML templates
│   └── index.html
└── static/            # Static assets
    ├── css/
    │   └── style.css
    └── js/
        └── app.js
```

### Development Mode

Run the server in development mode with auto-reload:
```bash
python run_server.py --reload --log-level debug
```

## License

This project is part of the BCE prediction framework. Please refer to the main project license.

## Support

For issues and questions:
1. Check the troubleshooting section
2. Review the API documentation at `/docs`
3. Check server logs for error details
4. Ensure all dependencies are properly installed 