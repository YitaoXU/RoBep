import numpy as np

from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score, 
    average_precision_score, roc_auc_score, f1_score, 
    precision_score, recall_score, matthews_corrcoef,
    accuracy_score, confusion_matrix, roc_curve, precision_recall_curve
)

def calculate_graph_metrics(preds, labels, threshold=0.5):
    """
    Calculate graph-level metrics for recall prediction.
    
    Args:
        preds: Predicted recall values (numpy array)
        labels: True recall values (numpy array)  
        threshold: Threshold for binary classification (default: 0.5, was 0.7)
        
    Returns:
        Dictionary of metrics
    """
    # Check for NaN values and replace with zeros
    preds = np.nan_to_num(preds, nan=0.0, posinf=1.0, neginf=0.0)
    labels = np.nan_to_num(labels, nan=0.0, posinf=1.0, neginf=0.0)
    
    # Convert predictions to binary for classification metrics
    pred_binary = (preds > threshold).astype(int)
    label_binary = (labels > threshold).astype(int)
    
    metrics = {}
    
    # Classification metrics
    if len(np.unique(label_binary)) > 1:  # Check if both classes exist
        metrics['recall'] = recall_score(label_binary, pred_binary, zero_division=0)
        metrics['precision'] = precision_score(label_binary, pred_binary, zero_division=0)
        metrics['mcc'] = matthews_corrcoef(label_binary, pred_binary)
        metrics['f1'] = f1_score(label_binary, pred_binary, zero_division=0)
        metrics['accuracy'] = accuracy_score(label_binary, pred_binary)
    else:
        metrics['recall'] = 0.0
        metrics['precision'] = 0.0
        metrics['mcc'] = 0.0
        metrics['f1'] = 0.0
        metrics['accuracy'] = 0.0
    
    # Regression metrics
    metrics['mse'] = mean_squared_error(labels, preds)
    metrics['mae'] = mean_absolute_error(labels, preds)
    metrics['r2'] = r2_score(labels, preds)
    
    return metrics

def calculate_node_metrics(preds, labels, find_threshold=False, include_curves=False):
    """
    Calculate node-level metrics for epitope prediction.
    
    Args:
        preds: Predicted probabilities (numpy array)
        labels: True binary labels (numpy array)
        find_threshold: If True, find the threshold that maximizes F1 score
        include_curves: If True, include PR and ROC curves for visualization
        
    Returns:
        Dictionary of metrics including optimal threshold if find_threshold=True
    """
    # Check for NaN values and replace with zeros
    preds = np.nan_to_num(preds, nan=0.0, posinf=1.0, neginf=0.0)
    labels = np.nan_to_num(labels, nan=0.0, posinf=1.0, neginf=0.0)
    
    metrics = {}
    
    # Check if both classes exist
    if len(np.unique(labels)) > 1:
        # AUROC and AUPRC (threshold-independent metrics)
        try:
            metrics['auroc'] = roc_auc_score(labels, preds)
            metrics['auprc'] = average_precision_score(labels, preds)
            
            # Include curves for visualization if requested
            if include_curves:
                # Calculate PR curve
                precision_curve, recall_curve, _ = precision_recall_curve(labels, preds)
                metrics['pr_curve'] = {
                    'precision': precision_curve,
                    'recall': recall_curve
                }
                
                # Calculate ROC curve
                fpr, tpr, _ = roc_curve(labels, preds)
                metrics['roc_curve'] = {
                    'fpr': fpr,
                    'tpr': tpr
                }
            else:
                metrics['pr_curve'] = None
                metrics['roc_curve'] = None
                
        except:
            metrics['auroc'] = 0.0
            metrics['auprc'] = 0.0
            metrics['pr_curve'] = None
            metrics['roc_curve'] = None
        
        # Find optimal threshold if requested
        if find_threshold:
            best_threshold, best_mcc = find_optimal_threshold(preds, labels)
            metrics['best_threshold'] = best_threshold
            threshold = best_threshold
        else:
            threshold = 0.5
            metrics['best_threshold'] = 0.5
        
        # Binary classification metrics using the determined threshold
        pred_binary = (preds > threshold).astype(int)
        metrics['f1'] = f1_score(labels, pred_binary, zero_division=0)
        metrics['mcc'] = matthews_corrcoef(labels, pred_binary)
        metrics['precision'] = precision_score(labels, pred_binary, zero_division=0)
        metrics['recall'] = recall_score(labels, pred_binary, zero_division=0)
        metrics['accuracy'] = accuracy_score(labels, pred_binary)
        
        # Confusion matrix components
        try:
            tn, fp, fn, tp = confusion_matrix(labels, pred_binary).ravel()
            metrics['true_positives'] = int(tp)
            metrics['false_positives'] = int(fp)
            metrics['true_negatives'] = int(tn)
            metrics['false_negatives'] = int(fn)
        except:
            metrics['true_positives'] = 0
            metrics['false_positives'] = 0
            metrics['true_negatives'] = 0
            metrics['false_negatives'] = 0
        
        # Store the threshold used for these metrics
        metrics['threshold_used'] = threshold
        
    else:
        # All metrics are 0 if only one class exists
        metrics['auroc'] = 0.0
        metrics['auprc'] = 0.0
        metrics['f1'] = 0.0
        metrics['mcc'] = 0.0
        metrics['precision'] = 0.0
        metrics['recall'] = 0.0
        metrics['accuracy'] = 0.0
        metrics['best_threshold'] = 0.5
        metrics['threshold_used'] = 0.5
        metrics['true_positives'] = 0
        metrics['false_positives'] = 0
        metrics['true_negatives'] = 0
        metrics['false_negatives'] = 0
        metrics['pr_curve'] = None
        metrics['roc_curve'] = None
    
    return metrics

def find_optimal_threshold(preds, labels, num_thresholds=100):
    """
    Find the threshold that maximizes F1 score.
    
    Args:
        preds: Predicted probabilities (numpy array)
        labels: True binary labels (numpy array)
        num_thresholds: Number of thresholds to test
        
    Returns:
        Tuple of (best_threshold, best_f1_score)
    """
    # Generate threshold candidates
    thresholds = np.linspace(0.01, 0.99, num_thresholds)
    
    best_mcc = 0.0
    best_threshold = 0.5
    
    for threshold in thresholds:
        pred_binary = (preds > threshold).astype(int)
        mcc = matthews_corrcoef(labels, pred_binary)
        
        if mcc > best_mcc:
            best_mcc = mcc
            best_threshold = threshold
    
    return best_threshold, best_mcc
