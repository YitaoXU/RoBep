import os
import json
import time
import numpy as np
from tqdm import tqdm
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
from collections import defaultdict

import torch
from torch.amp import autocast, GradScaler

from ..loss import CLoss
from .metrics import calculate_graph_metrics, calculate_node_metrics
from .results import evaluate_model
from .constants import BASE_DIR
from ..data.data import create_data_loader
from ..model.RoBep import get_model, RoBep
from ..model.scheduler import get_scheduler

torch.set_num_threads(12)

def check_for_nan_inf(tensor, name="tensor"):
    """Check for NaN or Inf values in tensor and raise error if found."""
    if torch.isnan(tensor).any():
        raise ValueError(f"NaN detected in {name}")
    if torch.isinf(tensor).any():
        raise ValueError(f"Inf detected in {name}")

def format_loss_info(loss_dict, prefix=""):
    """
    Format loss information from loss_dict for display.
    
    Args:
        loss_dict: Dictionary containing loss components
        prefix: Prefix for the output string
        
    Returns:
        Formatted string with loss breakdown
    """
    main_info = f"{prefix}Total: {loss_dict['loss/total']:.4f}, " \
                f"Region: {loss_dict['loss/region']:.4f}, " \
                f"Node: {loss_dict['loss/node']:.4f}"
    
    additional_info = []
    if loss_dict.get('loss/cls', 0) > 0:
        additional_info.append(f"Cls: {loss_dict['loss/cls']:.4f}, Reg: {loss_dict['loss/reg']:.4f}")
    if loss_dict.get('loss/consistency', 0) > 0:
        additional_info.append(f"Consistency: {loss_dict['loss/consistency']:.4f}")
    
    if additional_info:
        return main_info + "\n" + prefix + "  " + ", ".join(additional_info)
    return main_info

class Trainer:
    """
    Trainer class for RoBep model with comprehensive training and evaluation.
    
    Features:
    - Early stopping with patience
    - Model checkpointing (best AUPRC and best F1)
    - Mixed precision training
    - Comprehensive metrics calculation
    - Learning rate scheduling
    """

    @staticmethod
    def convert_to_serializable(obj):
        """Convert numpy/torch types to Python native types for JSON serialization."""
        if isinstance(obj, dict):
            return {k: Trainer.convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [Trainer.convert_to_serializable(item) for item in obj]
        elif isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        elif hasattr(obj, 'item'):  # For torch tensors and numpy scalars
            return obj.item()
        else:
            return obj

    def __init__(self, args):
        """
        Initialize Trainer with datasets, model, and hyperparameters.
        
        Args:
            args: Argument namespace containing all training parameters
        """
        self.args = args
        
        # Set device
        self.device = torch.device(f"cuda:{args.device_id}" if torch.cuda.is_available() else "cpu")
        print(f"[INFO] Using device: {self.device}")
        
        # Create data loaders
        use_embeddings2 = False if args.encoder == "esmc" else True
        start_time = time.time()
        self.train_loader, self.test_loader = create_data_loader(
            radii=args.radii,
            batch_size=args.batch_size,
            undersample=args.undersample,
            zero_ratio=args.zero_ratio,
            seed=args.seed,
            use_embeddings2=use_embeddings2
        )
        self.val_loader = self.test_loader
        end_time = time.time()
        print(f"[INFO] Data loading time: {end_time - start_time:.2f} seconds")
        
        print(f"[INFO] Train samples: {len(self.train_loader.dataset)}")
        print(f"[INFO] Test samples: {len(self.test_loader.dataset)}")

        # Create directories
        timestamp = getattr(args, 'timestamp', None)
        if timestamp is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        self.results_dir = Path(f"{BASE_DIR}/results/RoBep/{timestamp}")
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.model_dir = Path(f"{BASE_DIR}/models/RoBep/{timestamp}")
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        # Model save paths
        self.best_auprc_model_path = Path(self.model_dir / "best_auprc_model.bin")
        self.best_mcc_model_path = Path(self.model_dir / "best_mcc_model.bin")
        self.last_model_path = Path(self.model_dir / "last_model.bin")
        
        # Initialize model using the flexible model loader
        self.model = get_model(args)
        
        # Store finetune configuration for later use
        self.is_finetune_mode = getattr(args, 'mode', 'train') == 'finetune'
        if self.is_finetune_mode:
            print("[INFO] Finetune mode enabled - will load pretrained model and freeze most parameters")
        
        # Initialize optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), 
            lr=args.lr, 
            weight_decay=args.weight_decay
        )
        
        # Initialize scheduler
        self.scheduler = get_scheduler(
            args=args,
            optimizer=self.optimizer,
            num_samples=len(self.train_loader.dataset)
        )
        
        # Initialize loss function
        self.criterion = CLoss(
            region_loss_type=args.region_loss_type,
            region_weight=args.region_weight,
            node_loss_type=args.node_loss_type,
            node_loss_weight=args.node_loss_weight,
            consistency_weight=args.consistency_weight,
            consistency_type=args.consistency_type,
            threshold=args.threshold,
            label_smoothing=args.label_smoothing,
            gradnorm=getattr(args, 'gradnorm', False),
            gradnorm_alpha=getattr(args, 'gradnorm_alpha', 1.5),
            **{k: v for k, v in vars(args).items() if k in ['alpha', 'pos_weight', 'reg_weight', 'gamma_high_cls', 'cls_type', 'regression_type', 'weight_mode']}
        )
        
        # GradNorm parameters
        self.gradnorm_enabled = getattr(args, 'gradnorm', False)
        self.gradnorm_update_freq = getattr(args, 'gradnorm_update_freq', 10)
        self.gradnorm_step_counter = 0
        
        if self.gradnorm_enabled:
            print(f"[INFO] GradNorm enabled with alpha={getattr(args, 'gradnorm_alpha', 1.5)}, update frequency={self.gradnorm_update_freq}")
            # Initialize optimizer for task weights
            if hasattr(self.criterion, 'log_w_region') and hasattr(self.criterion, 'log_w_node'):
                self.task_weight_optimizer = torch.optim.Adam([
                    self.criterion.log_w_region,
                    self.criterion.log_w_node
                ], lr=0.025)  # Higher learning rate for task weights
                print(f"[INFO] GradNorm task weight optimizer initialized")
            else:
                print(f"[WARNING] GradNorm enabled but task weight parameters not found in criterion")
                self.gradnorm_enabled = False
        
        # Mixed precision setup
        self.mixed_precision = args.mixed_precision
        if self.mixed_precision:
            self.scaler = GradScaler('cuda')
            print("[INFO] Using mixed precision training")
        else:
            self.scaler = None
            print("[INFO] Using full precision training")
        
        # Training parameters
        self.patience = args.patience
        self.threshold = args.threshold
        self.log_steps = 50
        self.max_grad_norm = getattr(args, 'max_grad_norm', 1.0)  # Gradient clipping threshold
                 
        # Training history
        self.train_history = []
        self.val_history = []
        
        # Save configuration
        if not self.is_finetune_mode:
            self.save_config()
            
        self.best_threshold = None

    def train(self):
        """Train the model with early stopping and checkpointing."""
        print("[INFO] Starting training...")
        
        self.model.to(self.device)
        # Also move loss function to device (important for GradNorm parameters)
        self.criterion.to(self.device)
                
        # Print model info
        self.model.print_param_count()
        
        # Initialize tracking variables
        best_val_auprc = 0.0
        best_val_mcc = 0.0
        patience_counter = -5 # 5 epochs avoid stopping too early
        start_time = time.time()
        
        for epoch in range(self.args.num_epoch):
            print(f"\n=== Epoch {epoch + 1}/{self.args.num_epoch} ===")
            
            # Training phase
            train_metrics = self._train_epoch(epoch)
            self.train_history.append(train_metrics)
            
            # Validation phase
            if self.val_loader:
                val_metrics = self._validate_epoch(epoch)
                self.val_history.append(val_metrics)
                
                # Check for improvement
                current_auprc = val_metrics['node_auprc']
                current_mcc = val_metrics['graph_mcc']
                
                improved = False
                
                # Save best AUPRC model
                if current_auprc > best_val_auprc:
                    best_val_auprc = current_auprc
                    print(f"[INFO] New best AUPRC: {best_val_auprc:.4f} - Model saved")
                    self.model.save(self.best_auprc_model_path, threshold=val_metrics['threshold'])
                    improved = True
                
                # Save best F1 model
                if current_mcc > best_val_mcc:
                    best_val_mcc = current_mcc
                    print(f"[INFO] New best MCC: {best_val_mcc:.4f} - Model saved")
                    self.model.save(self.best_mcc_model_path, threshold=val_metrics['threshold'])
                    self.best_threshold = val_metrics['threshold']
                    improved = True
                
                # Early stopping logic
                if improved:
                    patience_counter = 0
                else:
                    patience_counter += 1
                    print(f"[INFO] No improvement for {patience_counter}/{self.patience} epochs")
                
                if patience_counter >= self.patience:
                    print(f"[INFO] Early stopping triggered after {epoch + 1} epochs")
                    break
            
            # Save last model
            self.model.save(self.last_model_path)
        
        end_time = time.time()
        print(f"\n[INFO] Training completed for {end_time - start_time:.2f} seconds")
        print(f"[INFO] Best validation AUPRC: {best_val_auprc:.4f}")
        print(f"[INFO] Best validation MCC: {best_val_mcc:.4f}")
        
        # Save training history
        self._save_training_history()
        
        # Evaluate the model
        print("\n" + "="*80)
        print("[INFO] Evaluating best AUPRC model...")
        results = evaluate_model(
            model_path=self.best_auprc_model_path,
            device_id=self.args.device_id,
            radius=self.args.radius,
            threshold=self.best_threshold,
            k=self.args.k,
            verbose=True,
            split="test",
            encoder=self.args.encoder
        )
        
        print("="*80)
        print("\n[INFO] Evaluating best MCC model...")
        results = evaluate_model(
            model_path=self.best_mcc_model_path,
            device_id=self.args.device_id,
            radius=self.args.radius,
            threshold=self.best_threshold,
            k=self.args.k,
            verbose=True,
            split="test",
            encoder=self.args.encoder
        )
    
    def finetune(self, pretrained_model_path=None):
        """
        Finetune the node prediction parameters of the model.
        
        Args:
            pretrained_model_path: Path to pretrained model. If None, uses best_auprc_model_path
        """
        print("[INFO] Starting finetuning...")
        
        # Load pretrained model
        if pretrained_model_path is None:
            # Prefer AUPRC model, fallback to MCC model
            if self.best_mcc_model_path.exists():
                pretrained_model_path = self.best_mcc_model_path
                print(f"[INFO] Using best MCC model for finetuning")
            elif self.best_auprc_model_path.exists():
                pretrained_model_path = self.best_auprc_model_path
                print(f"[INFO] Using best AUPRC model for finetuning")
            else:
                raise FileNotFoundError(
                    f"No pretrained model found. Please train the model first or provide a pretrained_model_path.\n"
                    f"Expected paths: {self.best_auprc_model_path} or {self.best_mcc_model_path}"
                )
        else:
            # Convert string path to Path object if needed
            pretrained_model_path = Path(pretrained_model_path)
        
        if not pretrained_model_path.exists():
            raise FileNotFoundError(f"Pretrained model not found at {pretrained_model_path}")
        
        print(f"[INFO] Loading pretrained model from {pretrained_model_path}")
        self.model, loaded_threshold = RoBep.load(pretrained_model_path, device=self.device)
        print(f"[INFO] Loaded model with threshold: {loaded_threshold}")
        
        # Freeze all parameters first
        for param in self.model.parameters():
            param.requires_grad = False
        
        # Unfreeze specific modules for node prediction
        trainable_modules = ['node_gate', 'node_classifier']
        trainable_params = []
        frozen_params = []
        
        print("[INFO] Setting trainable parameters:")
        for name, param in self.model.named_parameters():
            is_trainable = False
            for module_name in trainable_modules:
                if module_name in name:
                    param.requires_grad = True
                    is_trainable = True
                    trainable_params.append(param)
                    print(f"  - {name} (trainable)")
                    break
            
            if not is_trainable:
                frozen_params.append(param)
        
        print(f"[INFO] Trainable parameters: {len(trainable_params)}")
        print(f"[INFO] Frozen parameters: {len(frozen_params)}")
                
        # Print model info
        self.model.print_param_count()
        
        if len(trainable_params) == 0:
            raise ValueError("No trainable parameters found! Check trainable_modules list.")
        
        # Reinitialize optimizer with only trainable parameters
        self.optimizer = torch.optim.AdamW(
            trainable_params,
            lr=self.args.lr * 0.1,  # Use smaller learning rate for finetuning
            weight_decay=self.args.weight_decay
        )
        
        # Reinitialize scheduler
        self.args.schedule_type = "cosine"
        self.args.warmup_ratio = 0.0
        self.scheduler = get_scheduler(
            args=self.args,
            optimizer=self.optimizer,
            num_samples=len(self.train_loader.dataset)
        )
        
        # Adjust loss function weights for node-focused training
        print("[INFO] Adjusting loss weights for node-focused training")
        self.criterion.region_weight = 0.0
        self.criterion.node_loss_weight = 1.0
        self.gradnorm_enabled = False
        
        # Reset training history for finetuning
        self.train_history = []
        self.val_history = []
        
        # Update model save paths for finetuning
        self.best_finetuned_model_path = self.model_dir / "best_finetuned_model.bin"
        self.last_finetuned_model_path = self.model_dir / "last_finetuned_model.bin"
        
        print("[INFO] Finetune setup completed. Use train() method to start finetuning.")
        
        # Start finetuning training loop
        self._finetune_train()
        
        print("\n" + "="*80)
        print("[INFO] Evaluating finetuned model...")
        results = evaluate_model(
            model_path=self.best_finetuned_model_path,
            device_id=self.args.device_id,
            radius=self.args.radius,
            k=self.args.k,
            verbose=True,
            split="test"
        )

    def _finetune_train(self):
        """Training loop specifically for finetuning."""
        print("[INFO] Starting finetune training loop...")
        
        self.model.to(self.device)
        # Also move loss function to device (important for GradNorm parameters)
        self.criterion.to(self.device)
        
        # Initialize tracking variables
        best_val_auprc = 0.0
        patience_counter = 0  # No initial patience for finetuning
        start_time = time.time()
        
        # Use fewer epochs for finetuning
        finetune_epochs = self.args.num_epoch  # Max 60 epochs for finetuning
        
        for epoch in range(finetune_epochs):
            print(f"\n=== Finetune Epoch {epoch + 1}/{finetune_epochs} ===")
            
            # Training phase
            train_metrics = self._train_epoch(epoch)
            self.train_history.append(train_metrics)
            
            # Validation phase
            if self.val_loader:
                val_metrics = self._validate_epoch(epoch)
                self.val_history.append(val_metrics)
                
                # Check for improvement (focus on node metrics)
                current_auprc = val_metrics['node_auprc']
                
                improved = False
                
                # Save best AUPRC model
                if current_auprc > best_val_auprc:
                    best_val_auprc = current_auprc
                    print(f"[INFO] New best AUPRC: {best_val_auprc:.4f} - Finetuned model saved")
                    self.model.save(self.best_finetuned_model_path, threshold=val_metrics['threshold'])
                    improved = True
                
                # Early stopping logic (shorter patience for finetuning)
                if improved:
                    patience_counter = 0
                else:
                    patience_counter += 1
                    print(f"[INFO] No improvement for {patience_counter}/{min(self.patience, 10)} epochs")
                
                if patience_counter >= min(self.patience, 10):  # Shorter patience
                    print(f"[INFO] Early stopping triggered after {epoch + 1} epochs")
                    break
            
            # Save last model
            self.model.save(self.last_finetuned_model_path)
        
        end_time = time.time()
        print(f"\n[INFO] Finetuning completed in {end_time - start_time:.2f} seconds")
        print(f"[INFO] Best validation AUPRC: {best_val_auprc:.4f}")
        
        # Save training history
        self._save_training_history()

    def _train_epoch(self, epoch):
        """Train for one epoch."""
        self.model.train()
        
        # Initialize accumulators for loss components and predictions
        loss_accumulators = defaultdict(float)
        all_global_preds = []
        all_global_labels = []
        all_node_preds = []
        all_node_labels = []
        
        pbar = tqdm(self.train_loader, desc=f"Training Epoch {epoch + 1}")
        
        for batch_idx, batch in enumerate(pbar):
            batch = batch.to(self.device)
            
            # Prepare targets
            targets = {
                'y': batch.y,
                'y_node': batch.y_node
            }
            
            # Forward pass
            if self.mixed_precision:
                with autocast('cuda'):
                    outputs = self.model(batch)
                    if self.gradnorm_enabled:
                        loss, loss_dict, individual_losses = self.criterion(outputs, targets, batch.batch, return_individual_losses=True)
                    else:
                        loss, loss_dict = self.criterion(outputs, targets, batch.batch)
                
                # Check for NaN/Inf in loss
                if torch.isnan(loss) or torch.isinf(loss):
                    # print(f"[WARNING] NaN/Inf loss detected at batch {batch_idx}, skipping batch")
                    continue
                
                # Backward pass and GradNorm update
                self.optimizer.zero_grad()
                
                # GradNorm update BEFORE main backward to preserve computation graph  
                if self.gradnorm_enabled and self.gradnorm_step_counter % self.gradnorm_update_freq == 0:
                    # Update GradNorm weights first
                    self.task_weight_optimizer.zero_grad()
                    gradnorm_info = self.criterion.update_gradnorm_weights(individual_losses, self.model)
                    if gradnorm_info:
                        loss_dict.update(gradnorm_info)
                    self.task_weight_optimizer.step()
                
                # Main backward pass
                self.scaler.scale(loss).backward()
                
                # Gradient clipping
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(batch)
                if self.gradnorm_enabled:
                    loss, loss_dict, individual_losses = self.criterion(outputs, targets, batch.batch, return_individual_losses=True)
                else:
                    loss, loss_dict = self.criterion(outputs, targets, batch.batch)
                
                # Check for NaN/Inf in loss
                if torch.isnan(loss) or torch.isinf(loss):
                    # print(f"[WARNING] NaN/Inf loss detected at batch {batch_idx}, skipping batch")
                    continue
                
                # Backward pass and GradNorm update
                self.optimizer.zero_grad()
                loss.backward(retain_graph=self.gradnorm_enabled)
                
                # GradNorm update BEFORE main backward to preserve computation graph
                if self.gradnorm_enabled and self.gradnorm_step_counter % self.gradnorm_update_freq == 0:
                    # Update GradNorm weights first
                    self.task_weight_optimizer.zero_grad()
                    gradnorm_info = self.criterion.update_gradnorm_weights(individual_losses, self.model)
                    if gradnorm_info:
                        loss_dict.update(gradnorm_info)
                    self.task_weight_optimizer.step()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                
                self.optimizer.step()
                if hasattr(self.scheduler, 'step'):
                    self.scheduler.step()
            
            # Update GradNorm step counter
            if self.gradnorm_enabled:
                self.gradnorm_step_counter += 1
            
            # Accumulate loss components from loss_dict
            for key, value in loss_dict.items():
                if key.startswith('loss/'):
                    loss_accumulators[key] += value
            
            # Extract predictions from loss_dict (already detached)
            global_preds = torch.sigmoid(loss_dict['logits/global']).cpu().numpy()
            global_labels = batch.y.detach().cpu().numpy()
            node_preds = torch.sigmoid(loss_dict['logits/node']).cpu().numpy()
            node_labels = batch.y_node.detach().cpu().numpy()
            
            all_global_preds.extend(global_preds)
            all_global_labels.extend(global_labels)
            all_node_preds.extend(node_preds)
            all_node_labels.extend(node_labels)
            
            # Update progress bar with detailed loss info
            pbar.set_postfix({
                'Total': f"{loss_dict['loss/total']:.4f}",
                'Region': f"{loss_dict['loss/region']:.4f}",
                'Node': f"{loss_dict['loss/node']:.4f}",
                'LR': f"{self.optimizer.param_groups[0]['lr']:.2e}",
                'RW': f"{loss_dict['loss/region_weight']:.3f}" if self.gradnorm_enabled else "",
                'NW': f"{loss_dict['loss/node_weight']:.3f}" if self.gradnorm_enabled else ""
            })
        
        # Calculate average loss components
        num_batches = len(self.train_loader)
        avg_losses = {key: value / num_batches for key, value in loss_accumulators.items()}
        
        # Convert to numpy arrays
        all_global_preds = np.array(all_global_preds)
        all_global_labels = np.array(all_global_labels)
        all_node_preds = np.array(all_node_preds)
        all_node_labels = np.array(all_node_labels)
        
        # Calculate metrics
        graph_metrics = calculate_graph_metrics(all_global_preds, all_global_labels, self.threshold)
        node_metrics = calculate_node_metrics(all_node_preds, all_node_labels, find_threshold=False)
        
        # Combine all metrics
        metrics = {
            'lr': self.optimizer.param_groups[0]['lr'],
            **avg_losses,  # Include all loss components
        }
        
        # Add prefixed metrics
        for k, v in graph_metrics.items():
            metrics[f'graph_{k}'] = v
        for k, v in node_metrics.items():
            metrics[f'node_{k}'] = v
        
        # Print epoch summary with detailed loss breakdown
        print(format_loss_info(avg_losses, "Train Loss - "))
        print(f"Train Graph MCC: {graph_metrics['mcc']:.4f}, F1: {graph_metrics['f1']:.4f}, Recall: {graph_metrics['recall']:.4f}, Precision: {graph_metrics['precision']:.4f}")
        print(f"Train Node AUPRC: {node_metrics['auprc']:.4f}, AUROC: {node_metrics['auroc']:.4f}, F1: {node_metrics['f1']:.4f} (threshold: {node_metrics['threshold_used']:.3f})")
        
        # Print GradNorm info if enabled
        if self.gradnorm_enabled:
            print(f"GradNorm Weights - Region: {avg_losses.get('loss/region_weight', 1.0):.3f}, Node: {avg_losses.get('loss/node_weight', 1.0):.3f}")
            if 'gradnorm/region_grad_norm' in avg_losses:
                print(f"GradNorm Info - Region GradNorm: {avg_losses.get('gradnorm/region_grad_norm', 0):.4f}, Node GradNorm: {avg_losses.get('gradnorm/node_grad_norm', 0):.4f}")
        
        return metrics

    def _validate_epoch(self, epoch):
        """Validate for one epoch."""
        self.model.eval()
        
        # Initialize accumulators for loss components and predictions
        loss_accumulators = defaultdict(float)
        all_global_preds = []
        all_global_labels = []
        all_node_preds = []
        all_node_labels = []
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc=f"Validation Epoch {epoch + 1}")
            
            for batch in pbar:
                batch = batch.to(self.device)
                
                # Prepare targets
                targets = {
                    'y': batch.y,
                    'y_node': batch.y_node
                }
                
                # Forward pass
                if self.mixed_precision:
                    with autocast('cuda'):
                        outputs = self.model(batch)
                        loss, loss_dict = self.criterion(outputs, targets, batch.batch)
                else:
                    outputs = self.model(batch)
                    loss, loss_dict = self.criterion(outputs, targets, batch.batch)
                
                # Accumulate loss components from loss_dict
                for key, value in loss_dict.items():
                    if key.startswith('loss/'):
                        loss_accumulators[key] += value
                
                # Extract predictions from loss_dict (already detached)
                global_preds = torch.sigmoid(loss_dict['logits/global']).cpu().numpy()
                global_labels = batch.y.detach().cpu().numpy()
                node_preds = torch.sigmoid(loss_dict['logits/node']).cpu().numpy()
                node_labels = batch.y_node.detach().cpu().numpy()
                
                all_global_preds.extend(global_preds)
                all_global_labels.extend(global_labels)
                all_node_preds.extend(node_preds)
                all_node_labels.extend(node_labels)
                
                # Update progress bar with detailed loss info
                pbar.set_postfix({
                    'Total': f"{loss_dict['loss/total']:.4f}",
                    'Region': f"{loss_dict['loss/region']:.4f}",
                    'Node': f"{loss_dict['loss/node']:.4f}"
                })
        
        # Calculate average loss components
        num_batches = len(self.val_loader)
        avg_losses = {key: value / num_batches for key, value in loss_accumulators.items()}
        
        # Convert to numpy arrays
        all_global_preds = np.array(all_global_preds)
        all_global_labels = np.array(all_global_labels)
        all_node_preds = np.array(all_node_preds)
        all_node_labels = np.array(all_node_labels)
        
        # Calculate metrics
        graph_metrics = calculate_graph_metrics(all_global_preds, all_global_labels, self.threshold)
        node_metrics = calculate_node_metrics(all_node_preds, all_node_labels, find_threshold=True)
        
        # Combine all metrics
        metrics = {
            'epoch': epoch,
            **avg_losses,  # Include all loss components
        }
        
        # Add prefixed metrics
        for k, v in graph_metrics.items():
            metrics[f'graph_{k}'] = v
        for k, v in node_metrics.items():
            metrics[f'node_{k}'] = v
        metrics['threshold'] = node_metrics['threshold_used']
        
        # Print epoch summary with detailed loss breakdown
        print(format_loss_info(avg_losses, "Val Loss - "))
        print(f"Val Graph MCC: {graph_metrics['mcc']:.4f}, F1: {graph_metrics['f1']:.4f}, Recall: {graph_metrics['recall']:.4f}, Precision: {graph_metrics['precision']:.4f}")
        print(f"Val Node AUPRC: {node_metrics['auprc']:.4f}, AUROC: {node_metrics['auroc']:.4f}, F1: {node_metrics['f1']:.4f}, MCC: {node_metrics['mcc']:.4f}, Precision: {node_metrics['precision']:.4f}, Recall: {node_metrics['recall']:.4f} (threshold: {node_metrics['threshold_used']:.3f})")
        
        return metrics

    def evaluate(self, model_path=None, split='test'):
        """
        Evaluate the model on test set.
        
        Args:
            model_path: Path to model checkpoint (if None, uses current model)
            split: Which split to evaluate ('test', 'val', 'train')
        """
        print(f"[INFO] Evaluating on {split} set...")
        
        # Load model if path provided
        if model_path:
            print(f"[INFO] Loading model from {model_path}")
            self.model, _ = RoBep.load(model_path, device=self.device)
        
        # Select data loader
        if split == 'test':
            data_loader = self.test_loader
        elif split == 'val' and self.val_loader:
            data_loader = self.val_loader
        elif split == 'train':
            data_loader = self.train_loader
        else:
            raise ValueError(f"Invalid split: {split}")
        
        self.model.eval()
        
        # Initialize accumulators for loss components and predictions
        loss_accumulators = defaultdict(float)
        all_global_preds = []
        all_global_labels = []
        all_node_preds = []
        all_node_labels = []
        
        with torch.no_grad():
            pbar = tqdm(data_loader, desc=f"Evaluating {split}")
            
            for batch in pbar:
                batch = batch.to(self.device)
                
                # Prepare targets
                targets = {
                    'y': batch.y,
                    'y_node': batch.y_node
                }
                
                # Forward pass
                if self.mixed_precision:
                    with autocast('cuda'):
                        outputs = self.model(batch)
                        loss, loss_dict = self.criterion(outputs, targets, batch.batch)
                else:
                    outputs = self.model(batch)
                    loss, loss_dict = self.criterion(outputs, targets, batch.batch)
                
                # Accumulate loss components from loss_dict
                for key, value in loss_dict.items():
                    if key.startswith('loss/'):
                        loss_accumulators[key] += value
                
                # Extract predictions from loss_dict (already detached)
                global_preds = torch.sigmoid(loss_dict['logits/global']).cpu().numpy()
                global_labels = batch.y.detach().cpu().numpy()
                node_preds = torch.sigmoid(loss_dict['logits/node']).cpu().numpy()
                node_labels = batch.y_node.detach().cpu().numpy()
                
                all_global_preds.extend(global_preds)
                all_global_labels.extend(global_labels)
                all_node_preds.extend(node_preds)
                all_node_labels.extend(node_labels)
        
        # Calculate average loss components
        num_batches = len(data_loader)
        avg_losses = {key: value / num_batches for key, value in loss_accumulators.items()}
        
        # Convert to numpy arrays
        all_global_preds = np.array(all_global_preds)
        all_global_labels = np.array(all_global_labels)
        all_node_preds = np.array(all_node_preds)
        all_node_labels = np.array(all_node_labels)
        
        # Calculate metrics (use optimal threshold for evaluation)
        graph_metrics = calculate_graph_metrics(all_global_preds, all_global_labels, self.threshold)
        node_metrics = calculate_node_metrics(all_node_preds, all_node_labels, find_threshold=True)
        
        # Print results with detailed loss breakdown
        print(f"\n=== {split.upper()} RESULTS ===")
        print(format_loss_info(avg_losses, "Loss - "))
        
        print(f"\nGraph-level Metrics (Recall Prediction):")
        print(f"  MCC: {graph_metrics['mcc']:.4f}")
        print(f"  F1: {graph_metrics['f1']:.4f}")
        print(f"  Recall: {graph_metrics['recall']:.4f}")
        print(f"  Precision: {graph_metrics['precision']:.4f}")
        print(f"  MSE: {graph_metrics['mse']:.4f}")
        print(f"  MAE: {graph_metrics['mae']:.4f}")
        print(f"  R²: {graph_metrics['r2']:.4f}")
        print(f"  Pearson r: {graph_metrics['pearson_r']:.4f}")
        
        print(f"\nNode-level Metrics (Epitope Prediction):")
        print(f"  AUPRC: {node_metrics['auprc']:.4f}")
        print(f"  AUROC: {node_metrics['auroc']:.4f}")
        print(f"  F1: {node_metrics['f1']:.4f} (optimal threshold: {node_metrics['best_threshold']:.3f})")
        print(f"  MCC: {node_metrics['mcc']:.4f}")
        print(f"  Precision: {node_metrics['precision']:.4f}")
        print(f"  Recall: {node_metrics['recall']:.4f}")
        print("=" * 40)
        
        # Save results with detailed loss components
        results = {
            'split': split,
            'loss_components': self.convert_to_serializable(avg_losses),
            'graph_metrics': self.convert_to_serializable(graph_metrics),
            'node_metrics': self.convert_to_serializable(node_metrics),
            'model_path': str(model_path) if model_path else None
        }
        
        results_file = self.results_dir / f"{split}_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        return results

    def _save_training_history(self):
        """Save training history to file and generate loss plots."""
        history = {
            'train_history': self.convert_to_serializable(self.train_history),
            'val_history': self.convert_to_serializable(self.val_history)
        }
        
        history_file = self.results_dir / "training_history.json"
        with open(history_file, 'w') as f:
            json.dump(history, f, indent=2)
        
        print(f"[INFO] Training history saved to {history_file}")
        
        # Generate loss plots
        self._plot_training_curves()

    def _plot_training_curves(self):
        """Generate and save training curves for losses."""
        if not self.train_history:
            print("[WARNING] No training history to plot")
            return
        
        # Extract epochs and loss data
        train_epochs = list(range(1, len(self.train_history) + 1))
        train_total_loss = [h.get('loss/total', 0) for h in self.train_history]
        train_region_loss = [h.get('loss/region', 0) for h in self.train_history]
        train_node_loss = [h.get('loss/node', 0) for h in self.train_history]
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Training and Validation Loss Curves', fontsize=16, fontweight='bold')
        
        # Plot total loss
        axes[0, 0].plot(train_epochs, train_total_loss, 'b-', label='Train', linewidth=2, marker='o', markersize=3)
        if self.val_history:
            val_epochs = list(range(1, len(self.val_history) + 1))
            val_total_loss = [h.get('loss/total', 0) for h in self.val_history]
            axes[0, 0].plot(val_epochs, val_total_loss, 'r-', label='Validation', linewidth=2, marker='s', markersize=3)
        axes[0, 0].set_title('Total Loss', fontweight='bold')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot region loss (global loss)
        axes[0, 1].plot(train_epochs, train_region_loss, 'b-', label='Train', linewidth=2, marker='o', markersize=3)
        if self.val_history:
            val_region_loss = [h.get('loss/region', 0) for h in self.val_history]
            axes[0, 1].plot(val_epochs, val_region_loss, 'r-', label='Validation', linewidth=2, marker='s', markersize=3)
        axes[0, 1].set_title('Region Loss (Global)', fontweight='bold')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot node loss
        axes[1, 0].plot(train_epochs, train_node_loss, 'b-', label='Train', linewidth=2, marker='o', markersize=3)
        if self.val_history:
            val_node_loss = [h.get('loss/node', 0) for h in self.val_history]
            axes[1, 0].plot(val_epochs, val_node_loss, 'r-', label='Validation', linewidth=2, marker='s', markersize=3)
        axes[1, 0].set_title('Node Loss', fontweight='bold')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Loss')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot combined comparison
        axes[1, 1].plot(train_epochs, train_total_loss, 'b-', label='Train Total', linewidth=2, alpha=0.8)
        axes[1, 1].plot(train_epochs, train_region_loss, 'g-', label='Train Region', linewidth=2, alpha=0.8)
        axes[1, 1].plot(train_epochs, train_node_loss, 'orange', label='Train Node', linewidth=2, alpha=0.8)
        if self.val_history:
            axes[1, 1].plot(val_epochs, val_total_loss, 'r--', label='Val Total', linewidth=2, alpha=0.8)
            axes[1, 1].plot(val_epochs, val_region_loss, 'cyan', linestyle='--', label='Val Region', linewidth=2, alpha=0.8)
            axes[1, 1].plot(val_epochs, val_node_loss, 'magenta', linestyle='--', label='Val Node', linewidth=2, alpha=0.8)
        axes[1, 1].set_title('All Losses Comparison', fontweight='bold')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Loss')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        # Adjust layout
        plt.tight_layout()
        
        # Save the plot
        plot_file = self.results_dir / "training_loss_curves.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"[INFO] Training loss curves saved to {plot_file}")
        
        # Also create individual plots for better readability
        self._plot_individual_loss_curves()

    def _plot_individual_loss_curves(self):
        """Generate individual loss curve plots for each loss type."""
        if not self.train_history:
            return
        
        # Extract epochs and loss data
        train_epochs = list(range(1, len(self.train_history) + 1))
        loss_types = [
            ('total', 'Total Loss'),
            ('region', 'Region Loss (Global)'),
            ('node', 'Node Loss')
        ]
        
        for loss_key, loss_title in loss_types:
            fig, ax = plt.subplots(1, 1, figsize=(10, 6))
            
            # Training loss
            train_loss = [h.get(f'loss/{loss_key}', 0) for h in self.train_history]
            ax.plot(train_epochs, train_loss, 'b-', label='Training', linewidth=2.5, marker='o', markersize=4)
            
            # Validation loss
            if self.val_history:
                val_epochs = list(range(1, len(self.val_history) + 1))
                val_loss = [h.get(f'loss/{loss_key}', 0) for h in self.val_history]
                ax.plot(val_epochs, val_loss, 'r-', label='Validation', linewidth=2.5, marker='s', markersize=4)
            
            # Formatting
            ax.set_title(f'{loss_title} Over Training', fontsize=14, fontweight='bold')
            ax.set_xlabel('Epoch', fontsize=12)
            ax.set_ylabel('Loss', fontsize=12)
            ax.legend(fontsize=11)
            ax.grid(True, alpha=0.3)
            
            # Add some styling
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_linewidth(0.5)
            ax.spines['bottom'].set_linewidth(0.5)
            
            # Save individual plot
            plot_file = self.results_dir / f"{loss_key}_loss_curve.png"
            plt.tight_layout()
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"[INFO] {loss_title} curve saved to {plot_file}")

    def save_config(self):
        """Save training configuration to JSON file."""
        config = {
            'model_config': self.model.get_config(),  # Use model's get_config method
            'training_config': {
                'num_epoch': self.args.num_epoch,
                'batch_size': self.args.batch_size,
                'lr': self.args.lr,
                'weight_decay': self.args.weight_decay,
                'patience': self.args.patience,
                'threshold': self.args.threshold,
                'mixed_precision': self.args.mixed_precision,
                'device_id': self.args.device_id
            },
            'data_config': {
                'radii': self.args.radii,
                'zero_ratio': self.args.zero_ratio,
                'undersample': self.args.undersample,
                'seed': self.args.seed
            },
            'loss_config': {
                'region_loss_type': self.args.region_loss_type,
                'reg_weight': self.args.reg_weight,
                'cls_type': self.args.cls_type,
                'gamma_high_cls': self.args.gamma_high_cls,
                'regression_type': self.args.regression_type,
                'node_loss_type': self.args.node_loss_type,
                'alpha': self.args.alpha,
                'gamma': self.args.gamma,
                'pos_weight': self.args.pos_weight,
                'node_loss_weight': self.args.node_loss_weight,
                'region_weight': self.args.region_weight,
                'consistency_weight': self.args.consistency_weight,
                'consistency_type': self.args.consistency_type,
                'label_smoothing': self.args.label_smoothing,
                'gradnorm': getattr(self.args, 'gradnorm', False),
                'gradnorm_alpha': getattr(self.args, 'gradnorm_alpha', 1.5),
                'gradnorm_update_freq': getattr(self.args, 'gradnorm_update_freq', 10)
            }
        }
        
        config_file = self.model_dir / "config.json"
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"[INFO] Configuration saved to {config_file}")
