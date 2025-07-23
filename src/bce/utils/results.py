import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import roc_curve, auc, precision_recall_curve

from .constants import BASE_DIR
from .loading import load_data_split
from ..antigen.antigen import AntigenChain
from .metrics import find_optimal_threshold, calculate_node_metrics

def evaluate_model(model_path, device_id=0, radius=18.0, threshold=0.5, k=5, 
                   verbose=True, split="test", save_results=True, output_dir=None, encoder="esmc"):
    """
    Evaluate RoBep model on a dataset split using both probability-based and voting-based predictions.
    
    Args:
        model_path: Path to the trained RoBep model
        device_id: GPU device ID
        radius: Radius for spherical regions
        threshold: Threshold for probability-based predictions
        k: Number of top regions to select
        verbose: Whether to print progress
        split: Dataset split to evaluate ('test', 'val', 'train')
        save_results: Whether to save detailed results to files
        output_dir: Directory to save results (if save_results=True)
        
    Returns:
        Dictionary containing evaluation metrics for both prediction methods
    """
    print(f"[INFO] Evaluating RoBep model from {model_path}")
    print(f"[INFO] Settings:")
    print(f"  Radius: {radius}")
    print(f"  K: {k}")
    print(f"  Split: {split}\n")

    antigens = load_data_split(split, verbose=verbose)
    
    # Collect data for all proteins
    all_true_labels = []
    all_predicted_probs = []
    all_voted_labels = []
    all_predicted_binary = []
    
    protein_results = []
    
    for pdb_id, chain_id in tqdm(antigens, desc=f"Evaluating RoBep on {split} set", disable=not verbose):
        try:
            antigen_chain = AntigenChain.from_pdb(chain_id=chain_id, id=pdb_id)
            results = antigen_chain.evaluate(
                model_path=model_path,
                device_id=device_id,
                radius=radius,
                threshold=threshold,
                k=k,
                verbose=False,
                encoder=encoder
            )
            
            # Get true epitope labels as binary array
            true_epitopes = antigen_chain.get_epitope_residue_numbers()
            true_binary = []
            predicted_probs = []
            voted_binary = []
            predicted_binary = []
            
            # Convert to aligned arrays based on residue numbers
            for idx in range(len(antigen_chain.residue_index)):
                residue_num = int(antigen_chain.residue_index[idx])
                
                # True label
                true_binary.append(1 if residue_num in true_epitopes else 0)
                
                # Predicted probability
                predicted_probs.append(results['predictions'].get(residue_num, 0))
                
                # Voted prediction
                voted_binary.append(1 if residue_num in results['voted_epitopes'] else 0)
                
                # Probability-based prediction
                predicted_binary.append(1 if residue_num in results['predicted_epitopes'] else 0)
            
            # Store for overall evaluation
            all_true_labels.extend(true_binary)
            all_predicted_probs.extend(predicted_probs)
            all_voted_labels.extend(voted_binary)
            all_predicted_binary.extend(predicted_binary)
            
            length = len(antigen_chain.sequence)
            species = antigen_chain.get_species()
            precision = results['predicted_precision']
            recall = results['predicted_recall']
            f1 = 2 * precision * recall / (precision + recall + 1e-10)
            
            # Calculate PR-AUC using true_binary and predicted_probs
            if len(set(true_binary)) > 1:  # Check if there are both positive and negative samples
                pr_precision, pr_recall, _ = precision_recall_curve(true_binary, predicted_probs)
                pr_auc = auc(pr_recall, pr_precision)
            else:
                pr_auc = 0.0  # Default value when all labels are the same
            
            # Store individual protein results
            protein_results.append({
                'pdb_id': pdb_id,
                'chain_id': chain_id,
                'length': length,
                'species': species,
                'predicted_precision': precision,
                'predicted_recall': recall,
                'predicted_f1': f1,
                'pr_auc': pr_auc,
                'voted_precision': results['voted_precision'],
                'voted_recall': results['voted_recall'],
                'num_residues': len(true_binary),
                'num_true_epitopes': sum(true_binary),
                'num_predicted_epitopes': sum(predicted_binary),
                'num_voted_epitopes': sum(voted_binary),
                'true_epitopes': true_binary,
                'predicted_probabilities': predicted_probs
            })
            
        except Exception as e:
            if verbose:
                print(f"[WARNING] Failed to evaluate {pdb_id}_{chain_id}: {str(e)}")
            continue
    
    # Convert to numpy arrays
    all_true_labels = np.array(all_true_labels)
    all_predicted_probs = np.array(all_predicted_probs)
    all_voted_labels = np.array(all_voted_labels)
    all_predicted_binary = np.array(all_predicted_binary)
    
    # Calculate metrics for probability-based predictions (includes both probability and binary metrics)
    prob_metrics = calculate_node_metrics(all_predicted_probs, all_true_labels, find_threshold=True, include_curves=True)
    
    # Calculate metrics for voting-based predictions (binary only)
    vote_metrics = calculate_node_metrics(all_voted_labels.astype(float), all_true_labels, find_threshold=False)
    
    # Calculate metrics for probability-based binary predictions using original threshold
    pred_metrics = calculate_node_metrics(all_predicted_binary.astype(float), all_true_labels, find_threshold=False)
    
    # Additional statistics for comprehensive evaluation
    prediction_stats = {
        'prob_based': {
            'total_predicted_positive': int(np.sum(all_predicted_binary)),
            'prediction_rate': float(np.mean(all_predicted_binary))
        },
        'vote_based': {
            'total_predicted_positive': int(np.sum(all_voted_labels)),
            'prediction_rate': float(np.mean(all_voted_labels))
        }
    }
    
    # Overall statistics
    overall_stats = {
        'num_proteins': len(protein_results),
        'total_residues': len(all_true_labels),
        'total_true_epitopes': int(np.sum(all_true_labels)),
        'epitope_ratio': float(np.mean(all_true_labels)),
        'avg_protein_size': np.mean([p['num_residues'] for p in protein_results]),
        'avg_epitopes_per_protein': np.mean([p['num_true_epitopes'] for p in protein_results]),
        'prediction_stats': prediction_stats
    }
    
    if verbose:
        print_evaluation_results(prob_metrics, vote_metrics, pred_metrics, overall_stats, threshold)
    
    # Prepare results dictionary
    results = {
        'probability_metrics': prob_metrics,
        'voted_metrics': vote_metrics,
        'predicted_metrics': pred_metrics,
        'overall_stats': overall_stats,
        'protein_results': protein_results,
        'threshold': threshold
    }
    
    if save_results:
        if output_dir is None:
            # Handle both string and Path objects
            from pathlib import Path
            model_path_obj = Path(model_path)
            timestamp = model_path_obj.parent.name
            model_name = model_path_obj.name.split("_")[1]
            output_dir = BASE_DIR / "results" / "RoBep" / timestamp
        save_evaluation_results(results, output_dir, model_name)
    
    return results



def print_evaluation_results(prob_metrics, vote_metrics, pred_metrics, overall_stats, threshold):
    """Print formatted evaluation results for both prediction modes."""
    print(f"\n{'='*80}")
    print(f"RoBep MODEL EVALUATION RESULTS")
    print(f"{'='*80}")
    
    print(f"\nOverall Statistics:")
    print(f"  Number of proteins: {overall_stats['num_proteins']}")
    print(f"  Total residues: {overall_stats['total_residues']:,}")
    print(f"  Total true epitopes: {overall_stats['total_true_epitopes']:,}")
    print(f"  Epitope ratio: {overall_stats['epitope_ratio']:.3f}")
    print(f"  Average protein size: {overall_stats['avg_protein_size']:.1f}")
    print(f"  Average epitopes per protein: {overall_stats['avg_epitopes_per_protein']:.1f}")
    
    print(f"\n{'-'*40}")
    print(f"PROBABILITY-BASED PREDICTIONS")
    print(f"{'-'*40}")
    print(f"Threshold: {prob_metrics['best_threshold']}")
    print(f"\nProbability Metrics:")
    print(f"  AUPRC: {prob_metrics['auprc']:.4f}")
    print(f"  AUROC: {prob_metrics['auroc']:.4f}")
    print(f"\nBinary Classification Metrics:")
    print(f"  Accuracy:  {prob_metrics['accuracy']:.4f}")
    print(f"  Precision: {prob_metrics['precision']:.4f}")
    print(f"  Recall:    {prob_metrics['recall']:.4f}")
    print(f"  F1-Score:  {prob_metrics['f1']:.4f}")
    print(f"  MCC:       {prob_metrics['mcc']:.4f}")
    print(f"\nConfusion Matrix:")
    print(f"  True Pos:  {prob_metrics['true_positives']:>6} | False Pos: {prob_metrics['false_positives']:>6}")
    print(f"  False Neg: {prob_metrics['false_negatives']:>6} | True Neg:  {prob_metrics['true_negatives']:>6}")
    
    print(f"\n{'-'*40}")
    print(f"VOTING-BASED PREDICTIONS")
    print(f"{'-'*40}")
    print(f"Binary Classification Metrics:")
    print(f"  Accuracy:  {vote_metrics['accuracy']:.4f}")
    print(f"  Precision: {vote_metrics['precision']:.4f}")
    print(f"  Recall:    {vote_metrics['recall']:.4f}")
    print(f"  F1-Score:  {vote_metrics['f1']:.4f}")
    print(f"  MCC:       {vote_metrics['mcc']:.4f}")
    print(f"\nConfusion Matrix:")
    print(f"  True Pos:  {vote_metrics['true_positives']:>6} | False Pos: {vote_metrics['false_positives']:>6}")
    print(f"  False Neg: {vote_metrics['false_negatives']:>6} | True Neg:  {vote_metrics['true_negatives']:>6}")
    
    print(f"\n{'-'*40}")
    print(f"COMPARISON SUMMARY")
    print(f"{'-'*40}")
    print(f"{'Metric':<12} {'Probability':<12} {'Voting':<12} {'Difference':<12}")
    print(f"{'-'*48}")
    print(f"{'Accuracy':<12} {prob_metrics['accuracy']:<12.4f} {vote_metrics['accuracy']:<12.4f} {prob_metrics['accuracy']-vote_metrics['accuracy']:<12.4f}")
    print(f"{'Precision':<12} {prob_metrics['precision']:<12.4f} {vote_metrics['precision']:<12.4f} {prob_metrics['precision']-vote_metrics['precision']:<12.4f}")
    print(f"{'Recall':<12} {prob_metrics['recall']:<12.4f} {vote_metrics['recall']:<12.4f} {prob_metrics['recall']-vote_metrics['recall']:<12.4f}")
    print(f"{'F1-Score':<12} {prob_metrics['f1']:<12.4f} {vote_metrics['f1']:<12.4f} {prob_metrics['f1']-vote_metrics['f1']:<12.4f}")
    print(f"{'MCC':<12} {prob_metrics['mcc']:<12.4f} {vote_metrics['mcc']:<12.4f} {prob_metrics['mcc']-vote_metrics['mcc']:<12.4f}")
    
    print(f"\n{'='*80}")
    

def save_evaluation_results(results, output_dir=None, prefix="evaluation"):
    """
    Save detailed evaluation results to files for further analysis.
    
    Args:
        results: Dictionary containing all evaluation results
        output_dir: Directory to save results
        prefix: Prefix for output files
    """
    import os
    import json
    
    if output_dir is None:
        output_dir = BASE_DIR / "results" / "evaluation"
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save overall results as JSON
    results_to_save = {
        'probability_metrics': results['probability_metrics'],
        'voted_metrics': results['voted_metrics'], 
        'predicted_metrics': results['predicted_metrics'],
        'overall_stats': results['overall_stats'],
        'threshold': results['threshold']
    }
    
    # Remove non-serializable items (curves)
    if 'pr_curve' in results_to_save['probability_metrics']:
        if results_to_save['probability_metrics']['pr_curve'] is not None:
            # Convert numpy arrays to lists for JSON serialization
            results_to_save['probability_metrics']['pr_curve'] = {
                'precision': results_to_save['probability_metrics']['pr_curve']['precision'].tolist(),
                'recall': results_to_save['probability_metrics']['pr_curve']['recall'].tolist()
            }
    
    if 'roc_curve' in results_to_save['probability_metrics']:
        if results_to_save['probability_metrics']['roc_curve'] is not None:
            results_to_save['probability_metrics']['roc_curve'] = {
                'fpr': results_to_save['probability_metrics']['roc_curve']['fpr'].tolist(),
                'tpr': results_to_save['probability_metrics']['roc_curve']['tpr'].tolist()
            }
    
    # Save main results
    with open(os.path.join(output_dir, f"{prefix}_results.json"), 'w') as f:
        json.dump(results_to_save, f, indent=2)
    
    # Save protein-level results as CSV
    if 'protein_results' in results:
        df = pd.DataFrame(results['protein_results'])
        df.to_csv(os.path.join(output_dir, f"{prefix}_protein_results.csv"), index=False)
    
    print(f"\nResults saved to {output_dir}/")
    print(f"  - {prefix}_results.json: Overall metrics")
    print(f"  - {prefix}_protein_results.csv: Per-protein results")
    