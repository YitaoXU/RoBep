#!/usr/bin/env python3
"""
Main training script for RoBep model.
This script provides comprehensive argument parsing and training functionality.
"""

import argparse
import torch
from pathlib import Path

from bce.utils.trainer import Trainer
from bce.utils.tools import set_seed
from bce.utils.results import evaluate_model

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train RoBep model for epitope prediction")
    
    # Basic arguments
    parser.add_argument("--device_id", type=int, default=0, help="GPU device ID to use")
    parser.add_argument("--mixed_precision", action="store_true", help="Use mixed precision training")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    
    # Mode
    parser.add_argument("--mode", type=str, default="train", choices=["train", "finetune", "eval", "k_optimization"], help="Mode to run the model")
    
    # Finetune arguments
    parser.add_argument("--finetune_model_path", type=str, default=None, help="Path to pretrained model for finetuning")
    parser.add_argument("--timestamp", type=str, default=None, help="Timestamp for experiment naming")
    
    # Training arguments
    parser.add_argument("--batch_size", type=int, default=64, help="Training batch size")
    parser.add_argument("--num_epoch", type=int, default=120, help="Number of training epochs")
    
    # Evaluation arguments
    parser.add_argument("--eval", action="store_true", help="Only run evaluation, skip training")
    parser.add_argument("--model_path", type=str, default=None, help="Path to model for evaluation")
    parser.add_argument("--radius", type=float, default=18.0, help="Radius for graph construction")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"], help="Split to evaluate on")
    parser.add_argument("--best_threshold", type=float, default=None, help="Threshold for binary classification metrics")
    parser.add_argument("--k", type=int, default=7, help="Number of models to average")
    parser.add_argument("--eval_antigens", type=str, nargs="+", default=None, help="Custom list of antigens to evaluate (format: pdb_chain, e.g., 1A0O_A 1BVK_B)")
    
    # K optimization arguments
    parser.add_argument("--k_list", type=int, nargs="+", default=[1, 2, 3, 4, 5, 6, 7], help="List of k values to test for optimization")
    parser.add_argument("--val_antigens_file", type=str, default=None, help="Path to file containing validation antigen IDs (one per line: pdb_chain)")
    parser.add_argument("--optimization_metric", type=str, default="mcc", choices=["mcc", "f1", "auprc"], help="Metric to optimize for k selection")
    
    # Data arguments
    parser.add_argument("--radii", type=int, nargs="+", default=[16, 18, 20], help="Spherical radii for graph construction")
    parser.add_argument("--val", action="store_true", help="Use separate validation set (otherwise combine train+val)")
    parser.add_argument("--undersample", type=float, default=0.5, help="Undersampling ratio for training data")
    parser.add_argument("--zero_ratio", type=float, default=0.3, help="Ratio to downsample graphs with recall=0 (0.3 means keep 30%)")
    parser.add_argument("--threshold", type=float, default=0.5, help="Threshold for binary classification metrics")
    
    # Optimizer arguments
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-5, help="Weight decay for optimizer")
    parser.add_argument("--patience", type=int, default=15, help="Early stopping patience")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Maximum gradient norm for clipping")
    
    # Scheduler arguments
    parser.add_argument("--scheduler_type", type=str, default="cosine_restart", choices=["cosine", "cosine_restart", "step", "exponential", "one_cycle"], help="Type of learning rate scheduler")
    parser.add_argument("--warmup_ratio", type=float, default=0.1, help="Warmup ratio for scheduler")
    parser.add_argument("--warmup_type", type=str, default="linear", choices=["linear", "exponential"], help="Type of warmup")
    parser.add_argument("--eta_min", type=float, default=1e-6, help="Minimum learning rate for cosine scheduler")
    parser.add_argument("--T_mult", type=int, default=2, help="T_mult parameter for cosine restart scheduler")
    parser.add_argument("--rounds", type=int, default=5, help="Number of rounds for cosine restart scheduler")
    parser.add_argument("--decay_steps", type=int, default=3, help="Number of decay steps for step scheduler")

    # Model architecture arguments
    parser.add_argument("--in_dim", type=int, default=2560, help="Input feature dimension (ESM embedding size)")
    parser.add_argument("--encoder", type=str, default="esmc", choices=["esmc", "esm2"], help="Encoder type")
    parser.add_argument("--node_dims", type=int, nargs="+", default=[512, 256, 256], help="Node feature dimensions for EGNN layers")
    parser.add_argument("--edge_dim", type=int, default=32, help="Edge feature dimension")
    parser.add_argument("--dropout", type=float, default=0.4, help="Dropout rate")
    parser.add_argument("--activation", type=str, default="gelu", choices=["relu", "leaky_relu", "gelu", "silu", "tanh"], help="Activation function")
    parser.add_argument("--coords_agg", type=str, default="mean", choices=["mean", "sum", "max"], help="Coordinate aggregation method")
    
    # Global predictor
    parser.add_argument("--pooling", type=str, default="attention", choices=["attention", "add"], help="Pooling method for global predictor")
    
    # Node classifier
    parser.add_argument("--fusion_type", type=str, default="concat", choices=["concat", "add"], help="Fusion type for node classifier")
    parser.add_argument("--node_layers", type=int, default=2, help="Number of layers for node classifier")
    parser.add_argument("--out_dropout", type=float, default=0.2, help="Dropout rate for output layer")
    
    # Loss function arguments
    parser.add_argument("--region_loss_type", type=str, default="mse", choices=["dual", "mse", "combined"], help="Type of loss for graph-level prediction")
    parser.add_argument("--node_loss_type", type=str, default="focal", choices=["focal", "bce"], help="Type of loss for node-level prediction")
    parser.add_argument("--node_loss_weight", type=float, default=0.5, help="Weight for node-level loss")
    parser.add_argument("--region_weight", type=float, default=1.0, help="Weight for region-level loss")
    parser.add_argument("--consistency_weight", type=float, default=0.3, help="Weight for consistency loss")
    parser.add_argument("--consistency_type", type=str, default="mse", choices=["none", "mse"], help="Type of consistency loss")
    parser.add_argument("--label_smoothing", type=float, default=0.1, help="Label smoothing factor")
    
    # Node-level loss arguments
    parser.add_argument("--alpha", type=float, default=2.0, help="Alpha parameter for weighted MSE loss")
    parser.add_argument("--gamma", type=float, default=2.0, help="Gamma parameter for focal loss")
    parser.add_argument("--pos_weight", type=float, default=8.0, help="Positive class weight for BCE loss")
    
    # Region-level loss arguments
    parser.add_argument("--reg_weight", type=float, default=10.0, help="Weight for regression loss")
    parser.add_argument("--gamma_high_cls", type=float, default=2.0, help="Gamma for high class loss")
    parser.add_argument("--cls_type", type=str, default="bce", choices=["bce", "kl"], help="Type of classification loss")
    parser.add_argument("--regression_type", type=str, default="smooth_l1", choices=["smooth_l1", "mse"], help="Type of regression loss")
    parser.add_argument("--weight_mode", type=str, default="exp", choices=["exp", "linear"], help="Weighting mode for MSE loss")
    
    # GradNorm arguments
    parser.add_argument("--gradnorm", action="store_true", default=True, help="Enable GradNorm for dynamic loss balancing")
    parser.add_argument("--no_gradnorm", action="store_false", dest="gradnorm", help="Disable GradNorm")
    parser.add_argument("--gradnorm_alpha", type=float, default=2.0, help="GradNorm balancing parameter (controls the strength of balancing)")
    parser.add_argument("--gradnorm_update_freq", type=int, default=10, help="Frequency of GradNorm weight updates (in batches)")
    
    return parser.parse_args()

def main():
    """Main training function."""
    # Parse arguments
    args = parse_args()
    
    set_seed(args.seed)
    
    if args.mode == "eval":
        results = evaluate_model(
            model_path=args.model_path,
            device_id=args.device_id,
            radius=args.radius,
            threshold=args.best_threshold,
            k=args.k,
            verbose=True,
            split=args.split,
            encoder=args.encoder,
        )

        print(f"AUPRC: {results['probability_metrics']['auprc']:.4f}")
        print(f"AUROC: {results['probability_metrics']['auroc']:.4f}")

        print(f"Voted F1: {results['voted_metrics']['f1']:.4f}")

        print(f"Total proteins: {results['overall_stats']['num_proteins']}")
    
    elif args.mode == "finetune":
        # Finetune
        print("[INFO] Finetuning...")
        if args.finetune_model_path is not None:
            args.timestamp = args.finetune_model_path.split("/")[-2]
        trainer = Trainer(args)
        trainer.finetune(args.finetune_model_path)
        
    elif args.mode == "train":
        # Training
        print("=" * 60)
        print("RoBep Model Training")
        print("=" * 60)
        print(f"Device: cuda:{args.device_id}" if torch.cuda.is_available() else "cpu")
        print(f"Batch size: {args.batch_size}")
        print(f"Learning rate: {args.lr}")
        print(f"Epochs: {args.num_epoch}")
        print(f"Patience: {args.patience}")
        print(f"Node dimensions: {args.node_dims}")
        print(f"Radii: {args.radii}")
        print(f"Mixed precision: {args.mixed_precision}")
        print(f"Validation split: {'Enabled (8:2 train/val)' if args.val else 'Disabled (use test as val)'}")
        if args.val:
            print(f"K optimization: Enabled (will test k=[1,2,3,4,5,6,7] on validation set)")
        else:
            print(f"K optimization: Disabled (using default k={args.k})")
        if args.region_loss_type == "combined":
            print(f"Region loss: {args.region_loss_type}, with {args.cls_type} classification and {args.regression_type} regression")
        else:
            print(f"Region loss: {args.region_loss_type}")
        print(f"Node loss: {args.node_loss_type}")
        print(f"Scheduler: {args.scheduler_type}")
        if args.gradnorm:
            print(f"GradNorm: Enabled (alpha={args.gradnorm_alpha}, update_freq={args.gradnorm_update_freq})")
        else:
            print(f"GradNorm: Disabled")
        print(f"Model Config:")
        print(f"Encoder: {args.encoder}")
        print("=" * 60)
        
        # Initialize trainer
        trainer = Trainer(args)
        
        print("[INFO] Starting training...")
        trainer.train()
        
        print(f"\n[INFO] Training completed!")
        print(f"[INFO] Models saved to: {trainer.model_dir}")
        print(f"[INFO] Results saved to: {trainer.results_dir}")
    else:
        raise ValueError(f"Invalid mode: {args.mode}")

if __name__ == "__main__":
    main()
    
    
