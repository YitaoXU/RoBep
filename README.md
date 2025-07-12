---
title: B-cell Epitope Prediction Server
emoji: 🧬
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
license: mit
---

# 🧬 B-cell Epitope Prediction Server

A web-based interface for B-cell epitope prediction using the ReCEP model, built with Gradio and deployed on Hugging Face Spaces.

## 🚀 Features

- **🧬 Protein Structure Input**: Support for PDB ID lookup or file upload
- **🤖 AI-Powered Prediction**: Uses the ReCEP model for epitope prediction
- **📊 Comprehensive Results**: Detailed prediction statistics and epitope analysis
- **💾 Export Options**: Download results in JSON, CSV, and 3D visualization formats
- **🎨 Interactive 3D Visualization**: Molecular structure viewer using 3Dmol.js
- **⚡ Real-time Processing**: Progress tracking and error handling

## 📋 How to Use

### 1. Input Protein Structure

Choose one of two input methods:

#### Option A: PDB ID
- Enter a 4-character PDB ID (e.g., "5I9Q")
- Specify the chain ID (e.g., "A")

#### Option B: Upload PDB File
- Upload a PDB structure file (.pdb or .ent format)
- Optionally specify a custom PDB ID
- Specify the chain ID

### 2. Configure Prediction Parameters

#### Basic Parameters:
- **Chain ID**: Target protein chain (default: A)

#### Advanced Parameters (Optional):
- **Radius**: Spherical region radius in Ångstroms (default: 19.0)
- **Top-k Regions**: Number of top regions to analyze (default: 7)
- **Encoder**: Protein encoder type (ESM-C or ESM-2, default: ESM-C)
- **Device Configuration**: CPU or GPU processing (default: CPU Only)
- **Threshold**: Custom prediction threshold (leave empty for auto)
- **Auto-cleanup**: Automatically delete generated files after prediction

### 3. View Results

The application provides:

#### Prediction Summary
- Protein information (PDB ID, chain, length, sequence)
- Prediction statistics (epitope count, coverage rate, etc.)
- Top-k region centers
- Predicted epitope residues
- Binding region residues

#### Download Options
- **JSON Results**: Complete prediction data with metadata
- **CSV Results**: Residue-level predictions for analysis
- **3D Visualization**: Interactive HTML file with 3Dmol.js viewer

### 4. 3D Visualization

The downloadable HTML file includes:
- **Display Modes**: 
  - Predicted Epitopes: Highlight predicted epitope residues
  - Probability Gradient: Color residues by prediction confidence
- **Representation Styles**: Cartoon, Surface, Stick, Sphere
- **Interactive Controls**: Rotate, zoom, pan, reset view, save image

## 🔧 Technical Details

### Model Information
- **ReCEP Model**: State-of-the-art B-cell epitope prediction model
- **Encoders**: ESM-C (recommended) and ESM-2 support
- **Processing**: Supports both CPU and GPU processing
- **Auto-cleanup**: Automatically manages temporary files

### Input Requirements
- **PDB ID**: 4-character alphanumeric code
- **Chain ID**: Single character (A-Z, 0-9)
- **File Format**: PDB or ENT files up to 50MB

### Output Formats
- **JSON**: Complete structured prediction data
- **CSV**: Tabular format for analysis
- **HTML**: Interactive 3D visualization

## 🛠️ Local Development

To run locally:

```bash
# Install dependencies
pip install -r requirements.txt

# Run the application
python app.py
```

The application will be available at `http://localhost:7860`

## 📚 Examples

### Example 1: Basic Prediction
1. Input PDB ID: `5I9Q`
2. Chain ID: `A`
3. Click "Predict Epitopes"

### Example 2: Advanced Parameters
1. Input PDB ID: `1FBI`
2. Chain ID: `A`
3. Set Radius: `20.0`
4. Set Top-k Regions: `5`
5. Select Encoder: `esmc`
6. Click "Predict Epitopes"

## 🎯 Performance

- **Typical prediction time**: 2-5 minutes (depends on protein size and device)
- **Memory usage**: Varies based on protein size and device configuration
- **Supported protein sizes**: Up to 2046 residues (ESM model limit)

## 📄 Citation

If you use this application in your research, please cite:

```bibtex
@article{recep2024,
  title={ReCEP: Residue-Centric Epitope Prediction for B-cell Epitopes},
  author={[Your Research Team]},
  journal={[Journal Name]},
  year={2024}
}
```

## 🤝 Contributing

We welcome contributions! Please feel free to:
- Report issues
- Suggest improvements
- Submit pull requests

## 📧 Support

For questions or support:
- Create an issue in the repository
- Check the troubleshooting section
- Review the technical documentation

## 📜 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- ESM team for the protein language models
- 3Dmol.js team for the molecular visualization library
- Gradio team for the web framework
- Hugging Face for hosting the Space

---

**Note**: This is a research tool for B-cell epitope prediction. Results should be validated through experimental methods for clinical or commercial applications.
