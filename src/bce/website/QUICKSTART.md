# BCE Prediction Web Server - Quick Start Guide

## 🚀 Quick Installation

1. **Navigate to the website directory:**
   ```bash
   cd src/bce/website
   ```

2. **Install required dependencies:**
   ```bash
   pip install fastapi uvicorn python-multipart jinja2 aiofiles
   ```

3. **Test the setup (optional):**
   ```bash
   python test_setup.py
   ```

4. **Start the server:**
   ```bash
   python run_server.py
   ```

5. **Open your browser and visit:**
   ```
   http://localhost:8000
   ```

## 🧬 Quick Test

Try predicting epitopes for a sample protein:

1. Select "PDB ID" input method
2. Enter PDB ID: `1FBI`
3. Set Chain ID: `A`
4. Click "Predict Epitopes"
5. Wait for the prediction to complete (2-5 minutes)
6. Explore the 3D visualization and results

## ⚙️ Configuration

### Environment Variables (Optional)
```bash
export BCE_DEVICE_ID=0          # GPU device ID
export BCE_PORT=8000            # Server port
export BCE_HOST=0.0.0.0         # Server host
export ESM_TOKEN="your_token"   # ESM API token
```

### Advanced Parameters
- **Radius**: Spherical region size (default: 19.0 Å)
- **Top-k**: Number of regions to analyze (default: 7)
- **Encoder**: Protein encoder (esmc/esm2)
- **Threshold**: Prediction threshold (auto if empty)

## 📁 Project Structure

```
src/bce/website/
├── main.py                 # FastAPI application
├── run_server.py          # Launch script
├── config.py              # Configuration settings
├── test_setup.py          # Setup verification
├── requirements.txt       # Dependencies
├── templates/
│   └── index.html         # Web interface
└── static/
    ├── css/style.css      # Styling
    └── js/app.js          # Frontend logic
```

## 🐳 Docker Alternative

```bash
# Build and run with Docker
docker-compose up --build

# Access the server
http://localhost:8000
```

## 🔧 Troubleshooting

### Common Issues

1. **Import Errors:**
   ```bash
   pip install fastapi uvicorn python-multipart
   ```

2. **CUDA Not Available:**
   - Check GPU drivers and CUDA installation
   - The server will fall back to CPU processing

3. **Port Already in Use:**
   ```bash
   python run_server.py --port 8001
   ```

4. **File Upload Issues:**
   - Ensure PDB files are valid (.pdb or .ent format)
   - Check file size (max 50MB)

### Getting Help

- Check server logs in the terminal
- Visit `/docs` for API documentation
- Run `python test_setup.py` to diagnose issues

## 🌐 API Usage

The server provides REST API endpoints:

```bash
# Health check
curl http://localhost:8000/health

# Submit prediction (example with curl)
curl -X POST "http://localhost:8000/predict" \
     -F "pdb_id=1FBI" \
     -F "chain_id=A"

# Check status
curl http://localhost:8000/status/{task_id}

# Get results
curl http://localhost:8000/result/{task_id}
```

## 📊 Features

- ✅ **PDB ID or File Upload**: Flexible protein input
- ✅ **Real-time Progress**: Live prediction status
- ✅ **3D Visualization**: Interactive molecular viewer
- ✅ **Multiple Display Modes**: Epitopes, probabilities, regions
- ✅ **Export Results**: JSON and CSV download
- ✅ **Responsive Design**: Works on desktop and mobile

## 🎯 Next Steps

1. **For Development:**
   ```bash
   python run_server.py --reload --log-level debug
   ```

2. **For Production:**
   - Configure environment variables
   - Use reverse proxy (nginx)
   - Set up SSL certificates
   - Use Redis for task queue

3. **For AWS Deployment:**
   - Use the Docker configuration
   - Consider ECS or EC2 deployment
   - Set up load balancer and auto-scaling

---

**Need help?** Check the full [README.md](README.md) for detailed documentation. 