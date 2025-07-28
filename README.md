---
title: B-cell Epitope Prediction Server
emoji: 🧬
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 4.44.1
app_file: app.py
pinned: false
license: mit
---

# 🧬 B-cell Epitope Prediction Server

A web-based interface for B-cell epitope prediction using the RoBep model.

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
- **Radius**: Spherical region radius in Ångstroms (default: 18.0)
- **Top-k Regions**: Number of top regions to analyze (default: 7)
- **Encoder**: Protein encoder type (ESM-C only now)
- **Device Configuration**: CPU or GPU processing (CPU Only now)
- **Threshold**: Custom prediction threshold (leave empty for auto, required)

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

## 📜 License

This project is licensed under the MIT License - see the LICENSE file for details.

**Note**: This is a research tool for B-cell epitope prediction. Results should be validated through experimental methods for clinical or commercial applications.
