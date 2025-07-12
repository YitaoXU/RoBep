// Global variables
let currentTaskId = null;
let currentResult = null;
let viewer = null;
let progressInterval = null;

// Initialize the application
$(document).ready(function() {
    initializeEventListeners();
    setupFormValidation();
});

// Event listeners setup
function initializeEventListeners() {
    // Form submission
    $('#predictionForm').on('submit', handleFormSubmission);
    
    // Input method toggle
    $('input[name="input_method"]').on('change', toggleInputMethod);
    
    // Advanced parameters toggle
    $('#show_advanced_btn').on('click', toggleAdvancedParams);
    
    // Visualization controls
    $('#updateVizBtn').on('click', updateVisualization);
    $('#resetView').on('click', resetView);
    $('#saveImage').on('click', saveImage);
    
    // Sphere selection controls
    $('#sphereCount').on('change', handleSphereCountChange);
    
    // Download buttons
    $('#downloadJSON').on('click', downloadJSON);
    $('#downloadCSV').on('click', downloadCSV);
    
    // Retry button
    $('#retryBtn').on('click', resetForm);
}

// Toggle between PDB ID and file upload
function toggleInputMethod() {
    const method = $('input[name="input_method"]:checked').val();
    if (method === 'pdb_id') {
        $('#pdb_id_group').show();
        $('#file_upload_group').hide();
        $('#pdb_id').prop('required', true);
        $('#pdb_file').prop('required', false);
    } else {
        $('#pdb_id_group').hide();
        $('#file_upload_group').show();
        $('#pdb_id').prop('required', false);
        $('#pdb_file').prop('required', true);
    }
}

// Toggle advanced parameters
function toggleAdvancedParams() {
    const $btn = $('#show_advanced_btn');
    const $params = $('#advanced_params');
    
    if ($params.is(':visible')) {
        $params.slideUp();
        $btn.removeClass('active').text('Show Advanced Parameters');
    } else {
        $params.slideDown();
        $btn.addClass('active').text('Hide Advanced Parameters');
    }
}

// Form validation setup
function setupFormValidation() {
    $('#pdb_id').on('input', function() {
        const value = $(this).val().toUpperCase();
        $(this).val(value);
        
        if (value.length > 0 && !/^[A-Z0-9]{1,4}$/.test(value)) {
            $(this).addClass('is-invalid');
        } else {
            $(this).removeClass('is-invalid');
        }
    });
    
    $('#chain_id').on('input', function() {
        const value = $(this).val().toUpperCase();
        $(this).val(value);
        
        if (value.length > 0 && !/^[A-Z]$/.test(value)) {
            $(this).addClass('is-invalid');
        } else {
            $(this).removeClass('is-invalid');
        }
    });
}

// Handle form submission
function handleFormSubmission(e) {
    e.preventDefault();
    
    if (!validateForm()) {
        return;
    }
    
    const formData = new FormData();
    const inputMethod = $('input[name="input_method"]:checked').val();
    
    // Add form fields
    if (inputMethod === 'pdb_id') {
        formData.append('pdb_id', $('#pdb_id').val().trim());
    } else {
        const fileInput = $('#pdb_file')[0];
        if (fileInput.files.length > 0) {
            formData.append('pdb_file', fileInput.files[0]);
        }
        if ($('#pdb_id').val().trim()) {
            formData.append('pdb_id', $('#pdb_id').val().trim());
        }
    }
    
    formData.append('chain_id', $('#chain_id').val().trim() || 'A');
    formData.append('radius', $('#radius').val() || '19.0');
    formData.append('k', $('#k').val() || '7');
    formData.append('encoder', $('#encoder').val() || 'esmc');
    formData.append('device_id', $('#device_id').val() || '-1');
    
    if ($('#threshold').val()) {
        formData.append('threshold', $('#threshold').val());
    }
    
    // Submit prediction request
    submitPrediction(formData);
}

// Validate form data
function validateForm() {
    const inputMethod = $('input[name="input_method"]:checked').val();
    
    if (inputMethod === 'pdb_id') {
        const pdbId = $('#pdb_id').val().trim();
        if (!pdbId) {
            showError('Please enter a PDB ID');
            return false;
        }
        if (!/^[A-Z0-9]{4}$/.test(pdbId.toUpperCase())) {
            showError('PDB ID must be exactly 4 characters (letters and numbers)');
            return false;
        }
    } else {
        const fileInput = $('#pdb_file')[0];
        if (!fileInput.files.length) {
            showError('Please select a PDB file');
            return false;
        }
        
        const file = fileInput.files[0];
        if (!file.name.toLowerCase().endsWith('.pdb') && !file.name.toLowerCase().endsWith('.ent')) {
            showError('Please select a valid PDB file (.pdb or .ent)');
            return false;
        }
        
        if (file.size > 50 * 1024 * 1024) { // 50MB limit
            showError('File size must be less than 50MB');
            return false;
        }
    }
    
    const chainId = $('#chain_id').val().trim();
    if (!chainId) {
        showError('Please enter a chain ID');
        return false;
    }
    
    return true;
}

// Submit prediction request
function submitPrediction(formData) {
    // Disable form and show progress
    $('#predictBtn').prop('disabled', true).html('<span class="loading"></span>Starting prediction...');
    hideAllSections();
    
    $.ajax({
        url: '/predict',
        type: 'POST',
        data: formData,
        processData: false,
        contentType: false,
        success: function(response) {
            currentTaskId = response.task_id;
            showProgressSection();
            startProgressMonitoring();
        },
        error: function(xhr, status, error) {
            let errorMsg = 'Failed to start prediction';
            if (xhr.responseJSON && xhr.responseJSON.detail) {
                errorMsg = xhr.responseJSON.detail;
            }
            showError(errorMsg);
            resetFormButton();
        }
    });
}

// Start monitoring prediction progress
function startProgressMonitoring() {
    progressInterval = setInterval(checkProgress, 1000); // Check every 1 second for better sync
    checkProgress(); // Initial check
}

// Check prediction progress
function checkProgress() {
    if (!currentTaskId) return;
    
    $.ajax({
        url: `/status/${currentTaskId}`,
        type: 'GET',
        success: function(response) {
            updateProgress(response);
            
            if (response.status === 'completed') {
                clearInterval(progressInterval);
                loadResults();
            } else if (response.status === 'error' || response.status === 'failed') {
                clearInterval(progressInterval);
                showError(response.error || 'Prediction failed');
                resetFormButton();
            }
        },
        error: function() {
            clearInterval(progressInterval);
            showError('Failed to check prediction status');
            resetFormButton();
        }
    });
}

// Update progress display with smooth animation
function updateProgress(response) {
    const progress = response.progress || 0;
    const message = response.message || 'Processing...';
    
    // Animate progress bar smoothly
    $('#progressFill').animate({
        width: progress + '%'
    }, 300, 'swing');
    
    $('#progressMessage').text(message);
    
    // Add visual feedback for different stages
    if (progress < 30) {
        $('#progressFill').css('background', 'linear-gradient(90deg, #667eea 0%, #764ba2 100%)');
    } else if (progress < 80) {
        $('#progressFill').css('background', 'linear-gradient(90deg, #4ECDC4 0%, #44A08D 100%)');
    } else {
        $('#progressFill').css('background', 'linear-gradient(90deg, #56CCF2 0%, #2F80ED 100%)');
    }
}

// Load prediction results
function loadResults() {
    $.ajax({
        url: `/result/${currentTaskId}`,
        type: 'GET',
        success: function(data) {
            currentResult = data;
            displayResults(data);
            resetFormButton();
        },
        error: function() {
            showError('Failed to load prediction results');
            resetFormButton();
        }
    });
}

// Display prediction results
function displayResults(data) {
    hideAllSections();
    
    // Update protein information
    const proteinInfo = data.protein_info;
    $('#proteinId').text(proteinInfo.id);
    $('#proteinChain').text(proteinInfo.chain_id);
    $('#proteinLength').text(proteinInfo.length);
    
    // Update prediction summary
    const prediction = data.prediction;
    $('#epitopeCount').text(prediction.predicted_epitopes.length);
    $('#regionCount').text(prediction.top_k_regions.length);
    
    // Calculate coverage rate
    const coverageRate = ((prediction.top_k_region_residues.length / proteinInfo.length) * 100).toFixed(1);
    $('#coverageRate').text(coverageRate + '%');
    
    // Calculate mean region prediction value
    const regionValues = prediction.top_k_regions.map(region => region.predicted_value || 0);
    const meanRegion = regionValues.length > 0 ? (regionValues.reduce((a, b) => a + b, 0) / regionValues.length).toFixed(3) : '0.000';
    $('#meanRegion').text(meanRegion);
    
    // Get antigenicity (epitope_rate) from prediction
    const antigenicity = prediction.epitope_rate ? prediction.epitope_rate.toFixed(3) : '0.000';
    $('#antigenicity').text(antigenicity);
    
    // Display epitope residues
    displayEpitopeResidues(prediction.predicted_epitopes);
    
    // Display binding region residues
    displayBindingRegionResidues(prediction.top_k_region_residues);
    
    // Initialize 3D visualization
    initialize3DViewer(data.visualization);
    
    // Generate sphere checkboxes for custom selection
    generateSphereCheckboxes();
    
    // Show results section
    $('#resultsSection').addClass('fade-in').show();
}

// Display epitope residues list
function displayEpitopeResidues(epitopes) {
    const container = $('#epitopeResidues');
    container.empty();
    
    if (epitopes.length === 0) {
        container.html('<p style="color: #666; font-style: italic;">No epitope residues predicted</p>');
        return;
    }
    
    epitopes.sort((a, b) => a - b).forEach(residue => {
        const span = $('<span class="epitope-residue"></span>').text(residue);
        container.append(span);
    });
}

// Display binding region residues list
function displayBindingRegionResidues(bindingRegionResidues) {
    const container = $('#bindingRegionResidues');
    container.empty();
    
    if (!bindingRegionResidues || bindingRegionResidues.length === 0) {
        container.html('<p style="color: #666; font-style: italic;">No binding region residues found</p>');
        return;
    }
    
    // Remove duplicates and sort
    const uniqueResidues = [...new Set(bindingRegionResidues)].sort((a, b) => a - b);
    
    uniqueResidues.forEach(residue => {
        const span = $('<span class="binding-region-residue"></span>').text(residue);
        container.append(span);
    });
}

// Initialize 3D molecular viewer
function initialize3DViewer(vizData) {
    const element = $('#viewer3d')[0];
    viewer = $3Dmol.createViewer(element);
    
    // Add the protein structure
    viewer.addModel(vizData.pdb_data, 'pdb');
    
    // Set initial style
    updateVisualization();
}

// Update 3D visualization
function updateVisualization() {
    if (!viewer || !currentResult) return;
    
    const mode = $('#vizMode').val();
    const style = $('#vizStyle').val();
    const showSpheres = $('#showSpheres').is(':checked');
    const sphereCount = $('#sphereCount').val();
    
    // Clear EVERYTHING first - this fixes surface mode switching issues
    viewer.removeAllShapes();
    viewer.removeAllSurfaces();
    viewer.setStyle({}, {});
    
    const vizData = currentResult.visualization;
    const prediction = currentResult.prediction;
    
    // Base style configuration
    const baseStyle = {};
    if (style === 'surface') {
        // For surface mode, completely hide the cartoon to avoid interference
        baseStyle['cartoon'] = { hidden: true };
    } else {
        baseStyle[style] = { color: '#e6e6f7' };
    }
    viewer.setStyle({}, baseStyle);
    
    if (mode === 'prediction') {
        // Color predicted epitopes (skip in surface mode since we use surfaces for coloring)
        const epitopes = prediction.predicted_epitopes;
        if (epitopes.length > 0 && style !== 'surface') {
            const epitopeStyle = {};
            epitopeStyle[style] = { color: '#9C6ADE' };
            viewer.setStyle({ resi: epitopes }, epitopeStyle);
        }
        
    } else if (mode === 'probability') {
        // Color by probability gradient for ALL residues
        const predictions = prediction.predictions;
        
        // Get probability range for all residues
        const allProbs = Object.values(predictions).filter(p => p !== undefined);
        if (allProbs.length > 0) {
            const minProb = Math.min(...allProbs);
            const maxProb = Math.max(...allProbs);
            
            // Color all residues based on their probability (skip in surface mode)
            if (style !== 'surface') {
                Object.keys(predictions).forEach(residue => {
                    const prob = predictions[residue];
                    if (prob !== undefined) {
                        const normalizedProb = maxProb > minProb ? (prob - minProb) / (maxProb - minProb) : 0.5;
                        const color = interpolateColor('#E6F3FF', '#DC143C', normalizedProb);
                        
                        const probStyle = {};
                        probStyle[style] = { color: color };
                        viewer.setStyle({ resi: [parseInt(residue)] }, probStyle);
                    }
                });
            }
        }
        
    } else if (mode === 'regions') {
        // Color top-k regions with different colors
        const regions = prediction.top_k_regions;
        const colors = ['#FF6B6B', '#96CEB4', '#4ECDC4', '#45B7D1', '#FFEAA7', '#DDA0DD', '#87CEEB'];
        
        if (style !== 'surface') {
            regions.forEach((region, index) => {
                const color = colors[index % colors.length];
                const regionStyle = {};
                regionStyle[style] = { color: color };
                viewer.setStyle({ resi: region.covered_residues }, regionStyle);
            });
        }
    }
    
    // Add spherical regions if requested with WIREFRAME mode
    if (showSpheres && prediction.top_k_regions && prediction.top_k_regions.length > 0) {
        const regions = prediction.top_k_regions;
        const colors = ['#FF6B6B', '#96CEB4', '#4ECDC4', '#45B7D1', '#FFEAA7', '#DDA0DD', '#87CEEB'];
        
        // Determine which spheres to show
        let spheresToShow = [];
        if (sphereCount === 'custom') {
            const selectedIndices = getSelectedSphereIndices();
            spheresToShow = selectedIndices.map(idx => ({ region: regions[idx], index: idx }));
        } else {
            let numSpheres = sphereCount === 'all' ? regions.length : parseInt(sphereCount);
            numSpheres = Math.min(numSpheres, regions.length);
            spheresToShow = regions.slice(0, numSpheres).map((region, index) => ({ region, index }));
        }
        
        spheresToShow.forEach(({ region, index }) => {
            // Get center coordinates for the region
            if (region.center_residue && region.radius) {
                try {
                    // Get CA coordinates for the center residue - use model 0 explicitly
                    const model = viewer.getModel(0);
                    const centerResidues = model.selectedAtoms({ 
                        resi: region.center_residue,
                        atom: 'CA'
                    });
                    
                    if (centerResidues.length > 0) {
                        const centerAtom = centerResidues[0];
                        const centerCoords = { x: centerAtom.x, y: centerAtom.y, z: centerAtom.z };
                        const sphereColor = colors[index % colors.length];
                        
                        // Add WIREFRAME sphere like in antigen.py
                        viewer.addSphere({
                            center: centerCoords,
                            radius: region.radius || 19.0,
                            color: sphereColor,
                            wireframe: true,
                            linewidth: 2.0  // Thicker lines for better visibility
                        });
                        
                        // Add center point marker
                        viewer.addSphere({
                            center: centerCoords,
                            radius: 0.7,
                            color: '#FFD700',  // Gold color for center
                            wireframe: false
                        });
                        
                        console.log(`Added wireframe sphere ${index + 1} at residue ${region.center_residue} with color ${sphereColor}`);
                    } else {
                        console.warn(`No CA atom found for residue ${region.center_residue}`);
                    }
                } catch (error) {
                    console.error(`Error adding sphere for region ${index}:`, error);
                }
            }
        });
    }
    
    // Handle surface generation ONLY when surface style is selected
    if (style === 'surface') {
        if (mode === 'prediction') {
            // Add base surface with good visibility
            viewer.addSurface($3Dmol.SurfaceType.VDW, {
                opacity: 0.8,  // Increased base opacity for better visibility
                color: '#e6e6f7'
            });
            
            // Add colored surface for epitopes with full opacity
            const epitopes = prediction.predicted_epitopes;
            if (epitopes.length > 0) {
                viewer.addSurface($3Dmol.SurfaceType.VDW, {
                    opacity: 1.0,  // Full opacity for vivid colors
                    color: '#9C6ADE'
                }, { resi: epitopes });
            }
        } else if (mode === 'probability') {
            // For probability mode, add surface with color mapping
            const predictions = prediction.predictions;
            const allProbs = Object.values(predictions).filter(p => p !== undefined);
            
            if (allProbs.length > 0) {
                const minProb = Math.min(...allProbs);
                const maxProb = Math.max(...allProbs);
                
                // Add base surface for uncovered residues
                viewer.addSurface($3Dmol.SurfaceType.VDW, {
                    opacity: 0.7,  // Increased base opacity for better visibility
                    color: '#e6e6f7'
                });
                
                // Add colored surfaces for each residue with probability
                Object.keys(predictions).forEach(residue => {
                    const prob = predictions[residue];
                    if (prob !== undefined) {
                        const normalizedProb = maxProb > minProb ? (prob - minProb) / (maxProb - minProb) : 0.5;
                        const color = interpolateColor('#E6F3FF', '#DC143C', normalizedProb);
                        
                        viewer.addSurface($3Dmol.SurfaceType.VDW, {
                            opacity: 1.0,  // Full opacity for vivid colors
                            color: color
                        }, { resi: [parseInt(residue)] });
                    }
                });
            }
        } else if (mode === 'regions') {
            // Add base surface
            viewer.addSurface($3Dmol.SurfaceType.VDW, {
                opacity: 0.8,  // Increased base opacity for better visibility
                color: '#e6e6f7'
            });
            
            // Add colored surfaces for regions
            const regions = prediction.top_k_regions;
            const colors = ['#FF6B6B', '#96CEB4', '#4ECDC4', '#45B7D1', '#FFEAA7', '#DDA0DD', '#87CEEB'];
            
            regions.forEach((region, index) => {
                const color = colors[index % colors.length];
                viewer.addSurface($3Dmol.SurfaceType.VDW, {
                    opacity: 1.0,  // Full opacity for vivid colors
                    color: color
                }, { resi: region.covered_residues });
            });
        } else {
            // Default surface
            viewer.addSurface($3Dmol.SurfaceType.VDW, {
                opacity: 0.9,  // High opacity for clear visibility
                color: '#e6e6f7'
            });
        }
    }
    
    viewer.zoomTo();
    viewer.render();
}

// Color interpolation helper
function interpolateColor(color1, color2, factor) {
    const c1 = hexToRgb(color1);
    const c2 = hexToRgb(color2);
    
    const r = Math.round(c1.r + factor * (c2.r - c1.r));
    const g = Math.round(c1.g + factor * (c2.g - c1.g));
    const b = Math.round(c1.b + factor * (c2.b - c1.b));
    
    return rgbToHex(r, g, b);
}

function hexToRgb(hex) {
    const result = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex);
    return result ? {
        r: parseInt(result[1], 16),
        g: parseInt(result[2], 16),
        b: parseInt(result[3], 16)
    } : null;
}

function rgbToHex(r, g, b) {
    return "#" + ((1 << 24) + (r << 16) + (g << 8) + b).toString(16).slice(1);
}

// Reset 3D view
function resetView() {
    if (viewer) {
        viewer.zoomTo();
        viewer.render();
    }
}

// Save 3D image
function saveImage() {
    if (viewer) {
        viewer.pngURI(function(uri) {
            const link = document.createElement('a');
            link.href = uri;
            link.download = 'protein_structure.png';
            link.click();
        });
    }
}

// Download results as JSON
function downloadJSON() {
    if (!currentResult) return;
    
    const dataStr = JSON.stringify(currentResult, null, 2);
    const dataBlob = new Blob([dataStr], { type: 'application/json' });
    const url = URL.createObjectURL(dataBlob);
    
    const link = document.createElement('a');
    link.href = url;
    link.download = `epitope_prediction_${currentResult.protein_info.id}_${currentResult.protein_info.chain_id}.json`;
    link.click();
    
    URL.revokeObjectURL(url);
}

// Download results as CSV
function downloadCSV() {
    if (!currentResult) return;
    
    const prediction = currentResult.prediction;
    const proteinInfo = currentResult.protein_info;
    
    let csv = 'Residue,Probability,Is_Epitope\n';
    
    for (let i = 1; i <= proteinInfo.length; i++) {
        const prob = prediction.predictions[i] || 0;
        const isEpitope = prediction.predicted_epitopes.includes(i) ? 1 : 0;
        csv += `${i},${prob},${isEpitope}\n`;
    }
    
    const dataBlob = new Blob([csv], { type: 'text/csv' });
    const url = URL.createObjectURL(dataBlob);
    
    const link = document.createElement('a');
    link.href = url;
    link.download = `epitope_prediction_${proteinInfo.id}_${proteinInfo.chain_id}.csv`;
    link.click();
    
    URL.revokeObjectURL(url);
}

// Utility functions
function hideAllSections() {
    $('#progressSection, #resultsSection, #errorSection').hide();
}

function showProgressSection() {
    hideAllSections();
    $('#progressSection').addClass('fade-in').show();
}

function showError(message) {
    hideAllSections();
    $('#errorMessage').text(message);
    $('#errorSection').addClass('fade-in').show();
}

function resetForm() {
    // Clear current task
    if (progressInterval) {
        clearInterval(progressInterval);
        progressInterval = null;
    }
    currentTaskId = null;
    currentResult = null;
    
    // Reset form
    $('#predictionForm')[0].reset();
    $('#chain_id').val('A');
    $('#radius').val('19.0');
    $('#k').val('7');
    $('#encoder').val('esmc');
    $('#device_id').val('0');
    
    // Hide sections
    hideAllSections();
    
    // Reset button
    resetFormButton();
    
    // Reset input method
    $('input[name="input_method"][value="pdb_id"]').prop('checked', true);
    toggleInputMethod();
}

function resetFormButton() {
    $('#predictBtn').prop('disabled', false).text('Predict Epitopes');
}

// Handle sphere count selection change
function handleSphereCountChange() {
    const sphereCount = $('#sphereCount').val();
    
    if (sphereCount === 'custom') {
        $('#customSphereSelection').show();
        generateSphereCheckboxes();
    } else {
        $('#customSphereSelection').hide();
    }
}

// Generate sphere checkboxes for custom selection
function generateSphereCheckboxes() {
    if (!currentResult || !currentResult.prediction.top_k_regions) {
        return;
    }
    
    const regions = currentResult.prediction.top_k_regions;
    const container = $('#sphereCheckboxes');
    container.empty();
    
    regions.forEach((region, index) => {
        const sphereNum = index + 1;
        const checkboxId = `sphere_${sphereNum}`;
        
        const checkboxContainer = $('<div>')
            .addClass('sphere-checkbox')
            .attr('data-sphere', sphereNum);
        
        const checkbox = $('<input>')
            .attr('type', 'checkbox')
            .attr('id', checkboxId)
            .prop('checked', sphereNum <= 5); // Default: show first 5
        
        const label = $('<label>')
            .attr('for', checkboxId)
            .text(`Sphere ${sphereNum} (R${region.center_residue})`);
        
        checkboxContainer.append(checkbox, label);
        container.append(checkboxContainer);
        
        // Add click handler
        checkboxContainer.on('click', function(e) {
            if (e.target.type !== 'checkbox') {
                checkbox.prop('checked', !checkbox.prop('checked'));
            }
            
            if (checkbox.prop('checked')) {
                checkboxContainer.addClass('selected');
            } else {
                checkboxContainer.removeClass('selected');
            }
        });
        
        // Initialize visual state
        if (checkbox.prop('checked')) {
            checkboxContainer.addClass('selected');
        }
    });
}

// Get selected sphere indices for custom mode
function getSelectedSphereIndices() {
    const selected = [];
    $('#sphereCheckboxes input[type="checkbox"]:checked').each(function() {
        const sphereNum = parseInt($(this).attr('id').split('_')[1]);
        selected.push(sphereNum - 1); // Convert to 0-based index
    });
    return selected;
} 