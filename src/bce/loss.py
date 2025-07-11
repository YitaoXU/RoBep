import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# Loss function for region classification
class CombinedLoss(nn.Module):
    """
    Combined loss = classification loss (BCE or KL) + weighted MSE loss
    - Classification target is in [0, 1], allowing soft labels (e.g., label smoothing)
    - MSE is weighted according to target values (more emphasis on high targets)
    
    Args:
        cls_type (str): 'bce' or 'kl'
        reg_weight (float): Weight for MSE loss
        gamma_high_cls (float): Positive weight for BCE loss
        alpha (float): Strength of weighting in MSE
        mode (str): 'exp' or 'linear' weighting
        scale_weights (bool): Whether to normalize MSE weights to [1, 1+alpha]
        apply_sigmoid (bool): Whether to apply sigmoid to predictions
    """
    def __init__(self, 
                 cls_type: str = 'bce',
                 reg_weight: float = 5.0,
                 gamma_high_cls: float = 2.0,
                 alpha: float = 2.0,
                 weight_mode: str = 'exp',
                 scale_weights: bool = True,
                 apply_sigmoid: bool = True,
                 **kwargs):
        super().__init__()
        self.cls_type = cls_type
        self.reg_weight = reg_weight
        self.apply_sigmoid = apply_sigmoid
        self.eps = 1e-6

        if cls_type == 'bce':
            self.bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(gamma_high_cls))
        elif cls_type == 'kl':
            # No built-in KL for binary, use manual impl
            self.gamma = gamma_high_cls
        else:
            raise ValueError(f"Unsupported classification loss: {cls_type}")

        self.alpha = alpha
        self.weight_mode = weight_mode
        self.scale_weights = scale_weights

    def _get_mse_weights(self, target: torch.Tensor) -> torch.Tensor:
        target = torch.clamp(target, 0.0, 1.0)

        if self.weight_mode == "exp":
            exp_input = torch.clamp(self.alpha * target, max=5.0)
            raw_weights = torch.exp(exp_input)
            if self.scale_weights:
                max_weight = torch.exp(torch.tensor(min(self.alpha, 5.0), device=target.device))
                scaled = 1.0 + (raw_weights - 1.0) / (max_weight - 1.0 + self.eps) * self.alpha
                return torch.clamp(scaled, 1.0, 1.0 + self.alpha)
            return raw_weights

        elif self.weight_mode == "linear":
            return torch.clamp(1.0 + self.alpha * target, 1.0, 1.0 + self.alpha)
        
        else:
            raise ValueError(f"Invalid MSE weighting mode: {self.weight_mode}")

    def _kl_loss(self, pred_logits: torch.Tensor, target_probs: torch.Tensor) -> torch.Tensor:
        """
        KL(pred || target): assume target is a soft binary label ∈ (0,1),
        pred is logit, apply sigmoid internally.
        """
        pred_probs = torch.sigmoid(pred_logits)
        pred_probs = torch.clamp(pred_probs, self.eps, 1.0 - self.eps)
        target_probs = torch.clamp(target_probs, self.eps, 1.0 - self.eps)

        kl = target_probs * torch.log(target_probs / pred_probs) + \
             (1 - target_probs) * torch.log((1 - target_probs) / (1 - pred_probs))
        
        # Optional: upweight positive class
        weight = torch.ones_like(target_probs)
        weight[target_probs > 0.5] = self.gamma
        return (kl * weight).mean()

    def forward(self, pred: torch.Tensor, target: torch.Tensor):
        pred = torch.clamp(pred, -10.0, 10.0)
        target = torch.clamp(target, 0.0, 1.0)

        # --- Classification loss ---
        if self.cls_type == 'bce':
            cls_loss = self.bce(pred, target)
        elif self.cls_type == 'kl':
            cls_loss = self._kl_loss(pred, target)
        else:
            raise ValueError()

        if torch.isnan(cls_loss) or torch.isinf(cls_loss):
            cls_loss = torch.tensor(0.0, device=pred.device)

        # --- Weighted MSE loss ---
        pred_probs = torch.sigmoid(pred)
        weights = self._get_mse_weights(target)
        mse = ((pred_probs - target) ** 2) * weights
        reg_loss = mse.mean()

        if torch.isnan(reg_loss) or torch.isinf(reg_loss):
            reg_loss = torch.tensor(0.0, device=pred.device)

        # --- Total ---
        total_loss = cls_loss + self.reg_weight * reg_loss
        if torch.isnan(total_loss) or torch.isinf(total_loss):
            total_loss = torch.tensor(0.0, device=pred.device)

        return total_loss, cls_loss, reg_loss

class DualLoss(nn.Module):
    """
    Combined classification + regression loss for bounded [0,1] regression tasks.

    Components:
    1) Classification: Treat samples with target > threshold as positive (label=1), else 0.
       -> Binary cross entropy loss.
       -> Optionally apply stronger penalty (gamma_high_cls) on false negatives.

    2) Regression: For target > threshold, predict precise value.
       -> Smooth L1 or MSE loss.

    Final Loss = classification_loss + reg_weight * regression_loss
    """

    def __init__(self,
                 threshold: float = 0.7,
                 reg_weight: float = 10.0,
                 gamma_high_cls: float = 2.0,
                 regression_type: str = "smooth_l1",
                 **kwargs):
        super().__init__()
        self.threshold = threshold
        self.reg_weight = reg_weight
        self.gamma_high_cls = gamma_high_cls
        self.regression_type = regression_type.lower()

    def forward(self, pred: torch.Tensor, target: torch.Tensor):
        """
        Args:
            pred: (N,) raw logits
            target: (N,) target values in [0, 1]
        Returns:
            total_loss: scalar tensor
            cls_loss: classification component
            reg_loss: regression component
        """
        # Clamp inputs for numerical stability
        pred = torch.clamp(pred, -10.0, 10.0)
        target = torch.clamp(target, 0.0, 1.0)
        
        p = torch.sigmoid(pred)  # (N,) in [0,1]
        label_cls = (target > self.threshold).float()

        # --- Classification Loss ---
        bce_loss = F.binary_cross_entropy_with_logits(pred, label_cls, reduction='none')
        
        # Check for NaN/Inf in BCE loss
        bce_loss = torch.where(torch.isfinite(bce_loss), bce_loss, torch.zeros_like(bce_loss))

        # Optional: increase penalty for false negatives on important (target > threshold) samples
        weight = torch.ones_like(bce_loss)
        false_neg_mask = (label_cls == 1) & (p < self.threshold)
        weight[false_neg_mask] = self.gamma_high_cls

        cls_loss = (bce_loss * weight).mean()

        # --- Regression Loss (for high target values) ---
        reg_mask = (target > self.threshold)
        if reg_mask.any():
            if self.regression_type == "mse":
                reg_loss = F.mse_loss(p[reg_mask], target[reg_mask])
            else:
                reg_loss = F.smooth_l1_loss(p[reg_mask], target[reg_mask], beta=1.0)
            
            # Check for NaN/Inf in regression loss
            if torch.isnan(reg_loss) or torch.isinf(reg_loss):
                reg_loss = torch.tensor(0.0, device=pred.device)
        else:
            reg_loss = torch.tensor(0.0, device=pred.device)

        total_loss = cls_loss + self.reg_weight * reg_loss
        
        # Final check for NaN/Inf
        if torch.isnan(total_loss) or torch.isinf(total_loss):
            total_loss = torch.tensor(0.0, device=pred.device)
            
        return total_loss, cls_loss, reg_loss
    
class WeightedMSELoss(nn.Module):
    """
    Weighted Mean Squared Error Loss for [0, 1] targets, with dynamic target-based weights.

    This loss is useful when the target labels are in [0, 1] range and you want to emphasize 
    higher-valued targets more (e.g., binding probabilities, attention heatmaps, etc.).

    Args:
        alpha (float): Weight strength coefficient. Higher values give more emphasis to large targets.
                       Recommended range: 3.0–5.0.
        mode (str): Weighting mode. Options:
                    - 'exp': exponential weight (exp(alpha * target))
                    - 'linear': linear weight (1 + alpha * target)
        scale_weights (bool): Whether to scale weights to a consistent range [1, 1 + alpha]
                              for numerical stability (especially important for 'exp' mode).
        epsilon (float): Small constant to prevent numerical instability.
        apply_sigmoid (bool): Whether to apply sigmoid to predictions. Use if model output is unbounded.
    """
    def __init__(self, 
                 alpha: float = 3.0, 
                 weight_mode: str = 'exp', 
                 scale_weights: bool = True, 
                 epsilon: float = 1e-8,
                 apply_sigmoid: bool = True,
                 **kwargs):
        super().__init__()
        self.alpha = alpha
        self.weight_mode = weight_mode
        self.scale_weights = scale_weights
        self.epsilon = epsilon
        self.apply_sigmoid = apply_sigmoid

        if self.weight_mode not in ['exp', 'linear']:
            raise ValueError(f"Unsupported mode: {self.weight_mode}. Use 'exp' or 'linear'.")

    def _calc_weights(self, target: torch.Tensor) -> torch.Tensor:
        """
        Compute per-sample weights based on target values.

        Args:
            target (Tensor): Ground truth tensor with values in [0, 1].

        Returns:
            Tensor: Weight tensor with same shape as target.
        """
        # Clamp target to prevent extreme values
        target = torch.clamp(target, 0.0, 1.0)
        
        if self.weight_mode == 'exp':
            # Clamp alpha * target to prevent overflow
            exp_input = torch.clamp(self.alpha * target, max=5.0)  # exp(10) ≈ 22000
            raw_weights = torch.exp(exp_input)
        elif self.weight_mode == 'linear':
            raw_weights = 1.0 + self.alpha * target

        if self.scale_weights and self.weight_mode == 'exp':
            max_weight = torch.exp(torch.tensor(min(self.alpha, 5.0), device=target.device))
            # Linearly rescale to [1, 1 + alpha]
            scaled_weights = 1.0 + (raw_weights - 1.0) / (max_weight - 1.0 + self.epsilon) * self.alpha
            # Clamp final weights to reasonable range
            return torch.clamp(scaled_weights, 1.0, 1.0 + self.alpha)
        else:
            # Clamp weights to prevent extreme values
            return torch.clamp(raw_weights, 1.0, 100.0)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute the weighted MSE loss.

        Args:
            pred (Tensor): Predictions (either raw logits or [0, 1] values).
            target (Tensor): Ground truth targets in [0, 1].

        Returns:
            Tensor: Scalar loss.
        """
        if self.apply_sigmoid:
            # Clamp logits to prevent overflow in sigmoid
            pred = torch.clamp(pred, -10.0, 10.0)
            pred = torch.sigmoid(pred)
        
        # Clamp predictions to [0, 1] range
        pred = torch.clamp(pred, 0.0, 1.0)
        target = torch.clamp(target, 0.0, 1.0)

        base_loss = (pred - target) ** 2
        weights = self._calc_weights(target)
        weighted_loss = base_loss * weights
        
        # Check for NaN/Inf and replace with zero
        weighted_loss = torch.where(torch.isfinite(weighted_loss), weighted_loss, torch.zeros_like(weighted_loss))

        return weighted_loss.mean()

# Loss function for node classification
class FocalLoss(nn.Module):
    """
    Focal Loss for binary classification.
    Reduces the relative loss for well-classified examples, focusing more on hard examples.
    
    FL(pt) = -alpha * (1-pt)^gamma * log(pt)
    where pt is the probability of the target class.
    """
    def __init__(self, 
                 gamma=2.0, 
                 pos_weight=2.0, 
                 epsilon=1e-6,
                 **kwargs):  # Add **kwargs
        """
        Args:
            gamma (float): Focusing parameter. Reduces the loss contribution from easy examples.
                         Higher gamma means more focus on hard examples. (Default: 2.0)
            pos_weight (float): Weight for positive class to handle class imbalance. (Default: 2.0)
            epsilon (float): Small constant for numerical stability
        """
        super().__init__()
        self.gamma = gamma
        self.epsilon = epsilon
        self.register_buffer('pos_weight', torch.tensor(pos_weight))
        
    def forward(self, pred, target):
        p = torch.sigmoid(pred)
        p = torch.clamp(p, self.epsilon, 1 - self.epsilon)

        weight = torch.ones_like(target)
        weight[target == 1] = self.pos_weight

        pt = p * target + (1 - p) * (1 - target)
        focal_weight = (1 - pt) ** self.gamma

        # Manual BCE
        bce_loss = -target * torch.log(p) - (1 - target) * torch.log(1 - p)

        loss = focal_weight * weight * bce_loss
        return loss

class BCELoss(nn.Module):
    """Binary Cross Entropy Loss with optional pos_weight"""
    def __init__(self, 
                 pos_weight=1.0, 
                 epsilon=1e-6,
                 **kwargs):  # Add **kwargs
        """
        Args:
            pos_weight (float): Weight for positive class
            epsilon (float): Small constant for numerical stability
        """
        super().__init__()
        self.register_buffer('pos_weight', torch.tensor(pos_weight))
        self.epsilon = epsilon

    def forward(self, pred, target):
        return F.binary_cross_entropy_with_logits(
            pred, target,
            pos_weight=self.pos_weight,
            reduction='none'
        )

# First define base loss dictionary without SBEP
base_loss_dict = {
    "dual": DualLoss,
    "combined": CombinedLoss,
    "mse": WeightedMSELoss,
    "focal": FocalLoss,
    "bce": BCELoss
}

def get_base_loss(loss_type, **kwargs):
    """
    Factory function to create base loss instances
    Args:
        loss_type: str, type of loss to create
        **kwargs: parameters for the loss
    """
    if loss_type not in base_loss_dict:
        raise ValueError(f"Unknown loss type: {loss_type}. Available types: {list(base_loss_dict.keys())}")
    
    return base_loss_dict[loss_type](**kwargs)

class CLoss(nn.Module):
    def __init__(self,
                 region_loss_type="dual",
                 node_loss_type="bce",
                 node_loss_weight=0.5,
                 region_weight=1.0,
                 consistency_weight=0.1,
                 consistency_type="none",  # Options: ["none", "mse"]
                 threshold=0.7,
                 label_smoothing=0.0,
                 gradnorm=False,
                 gradnorm_alpha=1.5,  # GradNorm balancing parameter
                 **kwargs):
        super().__init__()

        self.node_loss_weight = node_loss_weight
        self.region_weight = region_weight
        self.consistency_weight = consistency_weight
        self.consistency_type = consistency_type
        self.label_smoothing = label_smoothing
        self.threshold = threshold
        self.gradnorm = gradnorm
        self.gradnorm_alpha = gradnorm_alpha

        # Task loss
        region_kwargs = kwargs.copy()
        if region_loss_type in ["dual", "combined"]:
            region_kwargs['threshold'] = threshold
        elif region_loss_type == "mse":
            region_kwargs.pop('threshold', None)

        node_kwargs = kwargs.copy()
        node_kwargs.pop('threshold', None)

        self.region_loss = get_base_loss(region_loss_type, **region_kwargs)
        self.node_loss = get_base_loss(node_loss_type, **node_kwargs)

        # For GradNorm
        if self.gradnorm:
            # Initialize task weights as learnable parameters
            self.log_w_region = nn.Parameter(torch.zeros(1))
            self.log_w_node = nn.Parameter(torch.tensor([math.log(self.node_loss_weight)], dtype=torch.float32))
            
            # Track initial losses and training progress
            self.register_buffer('initial_losses', None)
            self.register_buffer('step_count', torch.zeros(1))
            
            print(f"[INFO] GradNorm enabled with alpha={gradnorm_alpha}")

    def forward(self, outputs, targets, batch, return_individual_losses=False):
        device = outputs['global_pred'].device
        batch = batch.to(torch.long)

        # recall_targets = label_smoothing(targets['y'], self.label_smoothing)
        # node_targets = label_smoothing(targets['y_node'], self.label_smoothing)
        recall_targets = targets['y']
        node_targets = targets['y_node']

        # Region loss
        if isinstance(self.region_loss, (DualLoss, CombinedLoss)):
            region_loss, cls_loss, reg_loss = self.region_loss(outputs['global_pred'], recall_targets)
        else:
            region_loss = self.region_loss(outputs['global_pred'], recall_targets)
            cls_loss = torch.tensor(0.0, device=device)
            reg_loss = torch.tensor(0.0, device=device)

        # Node loss (mean over graph)
        node_loss_raw = self.node_loss(outputs['node_preds'], node_targets)
        graph_node_loss = torch.zeros(outputs['global_pred'].size(0), device=device)
        graph_node_loss.scatter_reduce_(
            dim=0, index=batch, src=node_loss_raw.float(), reduce="mean", include_self=False
        )
        node_loss = graph_node_loss.mean()

        # Consistency
        if self.consistency_type == "mse":
            # New: node → region consistency (node_avg should match global_pred)
            global_pred = torch.sigmoid(outputs['global_pred'])  # shape [B]
            node_probs = torch.sigmoid(outputs['node_preds'])    # shape [N]

            # Scatter mean: for each graph, average node probabilities
            sum_probs = torch.zeros_like(global_pred)
            count = torch.zeros_like(global_pred)
            sum_probs.scatter_add_(0, batch, node_probs)
            count.scatter_add_(0, batch, torch.ones_like(node_probs))
            node_mean = sum_probs / (count + 1e-8)

            # Supervise region prediction to match average of its node predictions
            consistency_loss = F.mse_loss(global_pred, node_mean)
        else:
            consistency_loss = torch.tensor(0.0, device=device)

        # Store individual losses for GradNorm
        individual_losses = {
            'region': region_loss,
            'node': node_loss
        }

        # Final loss calculation
        if self.gradnorm:
            # Get task weights and ensure they're on the correct device
            w_region = torch.exp(self.log_w_region).to(device)
            w_node = torch.exp(self.log_w_node).to(device)
            
            # Initialize losses on first step
            if self.initial_losses is None:
                self.initial_losses = torch.tensor([region_loss.item(), node_loss.item()], device=device)
                self.step_count.zero_()
            
            # Weighted total loss - ensure all components are on the same device
            consistency_term = torch.tensor(self.consistency_weight, device=device) * consistency_loss
            total_loss = w_region * region_loss + w_node * node_loss + consistency_term
            
            # Update step count
            self.step_count += 1
            
        else:
            # Use static weights - ensure all components are on the same device
            region_weight_tensor = torch.tensor(self.region_weight, device=device)
            node_weight_tensor = torch.tensor(self.node_loss_weight, device=device)
            consistency_weight_tensor = torch.tensor(self.consistency_weight, device=device)
            
            total_loss = (region_weight_tensor * region_loss + 
                         node_weight_tensor * node_loss + 
                         consistency_weight_tensor * consistency_loss)

        # Prepare return info
        loss_info = {
            "loss/total": total_loss.item(),
            "loss/region": region_loss.item(),
            "loss/node": node_loss.item(),
            "loss/cls": cls_loss.item(),
            "loss/reg": reg_loss.item(),
            "loss/consistency": consistency_loss.item(),
            "logits/global": outputs['global_pred'].detach(),
            "logits/node": outputs['node_preds'].detach()
        }
        
        if self.gradnorm:
            loss_info.update({
                "loss/region_weight": torch.exp(self.log_w_region).to(device).item(),
                "loss/node_weight": torch.exp(self.log_w_node).to(device).item(),
                "gradnorm/step_count": self.step_count.item()
            })
        else:
            loss_info.update({
                "loss/region_weight": self.region_weight,
                "loss/node_weight": self.node_loss_weight
            })

        if return_individual_losses:
            return total_loss, loss_info, individual_losses
        else:
            return total_loss, loss_info
    
    def update_gradnorm_weights(self, individual_losses, model):
        """
        Update GradNorm weights based on gradient norms and relative loss rates.
        This modifies the log task weights (log_w_region and log_w_node)
        such that the gradient magnitudes are balanced.

        Args:
            individual_losses (dict): Contains task-specific scalar losses.
            model (nn.Module): The full model.
        """
        if not self.gradnorm:
            return

        device = next(model.parameters()).device

        # Extract region and node losses
        region_loss = individual_losses['region']
        node_loss = individual_losses['node']

        # Initialize shared parameters (those used by both tasks)
        shared_params = []
        task_specific_modules = ['node_classifier', 'global_predictor', 'node_gate']
        for name, param in model.named_parameters():
            if not any(tsm in name for tsm in task_specific_modules) and param.requires_grad:
                shared_params.append(param)

        if len(shared_params) == 0:
            print("[WARNING] No shared parameters found for GradNorm update")
            return

        # Get current weights (exp of log weights)
        w_region = torch.exp(self.log_w_region).to(device)
        w_node = torch.exp(self.log_w_node).to(device)

        # On first step, record initial losses for relative loss rate computation
        if self.initial_losses is None:
            self.initial_losses = torch.tensor(
                [region_loss.item(), node_loss.item()],
                device=device
            )
            self.step_count.zero_()

        # Compute relative loss rates (used in target gradient norms)
        current_losses = torch.tensor(
            [region_loss.item(), node_loss.item()],
            device=device
        )
        if self.step_count.item() > 1:
            loss_ratios = current_losses / self.initial_losses
            relative_loss_rates = loss_ratios / loss_ratios.mean()
        else:
            relative_loss_rates = torch.ones(2, device=device)

        # Compute gradient norm for region task (weighted loss)
        region_grads = torch.autograd.grad(
            w_region * region_loss,
            shared_params,
            retain_graph=True,
            create_graph=True,
            allow_unused=True
        )
        region_grads = [g for g in region_grads if g is not None]
        if len(region_grads) == 0:
            print("[WARNING] No gradients found for region task")
            return
        region_grad_norm = torch.norm(torch.cat([g.flatten() for g in region_grads]))

        # Compute gradient norm for node task (weighted loss)
        node_grads = torch.autograd.grad(
            w_node * node_loss,
            shared_params,
            retain_graph=True,
            create_graph=True,
            allow_unused=True
        )
        node_grads = [g for g in node_grads if g is not None]
        if len(node_grads) == 0:
            print("[WARNING] No gradients found for node task")
            return
        node_grad_norm = torch.norm(torch.cat([g.flatten() for g in node_grads]))

        # Stack gradient norms into a single tensor
        grad_norms = torch.stack([region_grad_norm, node_grad_norm])
        avg_grad_norm = grad_norms.mean()

        # Compute target gradient norms using GradNorm equation
        target_grad_norms = avg_grad_norm * (relative_loss_rates ** self.gradnorm_alpha)
        target_grad_norms = target_grad_norms.detach()  # Stop gradient here

        # Compute GradNorm loss (L1 distance between actual and target grad norms)
        gradnorm_loss = F.l1_loss(grad_norms, target_grad_norms)

        # Backward the GradNorm loss to update log_w_*
        gradnorm_loss.backward()

        # Normalize weights so that their sum stays constant (e.g., 2)
        with torch.no_grad():
            total_weight = w_region + w_node
            self.log_w_region.data = torch.log(2 * w_region / total_weight)
            self.log_w_node.data = torch.log(2 * w_node / total_weight)

        # Step counter for GradNorm
        self.step_count += 1

        return {
            'gradnorm/region_grad_norm': region_grad_norm.item(),
            'gradnorm/node_grad_norm': node_grad_norm.item(),
            'gradnorm/avg_grad_norm': avg_grad_norm.item(),
            'gradnorm/region_target_norm': target_grad_norms[0].item(),
            'gradnorm/node_target_norm': target_grad_norms[1].item(),
            'gradnorm/relative_loss_rate_region': relative_loss_rates[0].item(),
            'gradnorm/relative_loss_rate_node': relative_loss_rates[1].item(),
            'gradnorm/gradnorm_loss': gradnorm_loss.item(),
            'loss/region_weight': w_region.item(),
            'loss/node_weight': w_node.item()
        }



def label_smoothing(target: torch.Tensor, smoothing: float = 0.1) -> torch.Tensor:
    """
    Applies label smoothing for binary classification targets.
    
    Args:
        target (Tensor): Tensor of shape (N,), containing 0 or 1.
        smoothing (float): Smoothing factor (between 0 and 1).
    
    Returns:
        Tensor: Smoothed labels where 0 -> 0.5 * smoothing, 1 -> 1 - 0.5 * smoothing
    """
    return target * (1.0 - smoothing) + 0.5 * smoothing

