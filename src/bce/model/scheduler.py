import math
from typing import Dict, Any, Union, Optional
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler, CosineAnnealingLR, StepLR, ExponentialLR, CosineAnnealingWarmRestarts, OneCycleLR


class AutoScheduler(_LRScheduler):
    """
    Automatic learning rate scheduler with warmup and configurable main schedule.
    """
    
    # Default parameters for different scheduler types
    DEFAULT_PARAMS = {
        "cosine": {"eta_min": 1e-6},
        "cosine_restart": {"T_mult": 2, "eta_min": 1e-6, "rounds": 5},
        "step": {"gamma": 0.5, "decay_steps": 3},
        "exponential": {"gamma": 0.95},
        "one_cycle": {"lr_mult": 10.0, "div_factor": 25.0, "final_div_factor": 1e4}
    }
    
    def __init__(
        self,
        optimizer: Optimizer,
        total_steps: int,
        scheduler_type: str = "cosine_restart",
        warmup_ratio: float = 0.1,
        warmup_type: str = "linear",
        **kwargs
    ):
        self.scheduler_type = scheduler_type
        self.warmup_type = warmup_type
        self.warmup_ratio = warmup_ratio
        self.total_steps = total_steps
        self.warmup_steps = max(1, int(total_steps * warmup_ratio))
        self.current_step = 0
        self._is_warmup = True
        
        # Merge default parameters with user-provided kwargs
        self.params = self._get_merged_params(kwargs)
        
        # Validate parameters
        self._validate_parameters()
        
        # Create the main scheduler BEFORE calling super().__init__
        # This is needed because super().__init__ will call step() immediately
        self.after_scheduler = None  # Initialize as None first
        
        # Initialize parent class
        super().__init__(optimizer)
        
        # Now create the main scheduler
        self.after_scheduler = self._create_main_scheduler()

    def _get_merged_params(self, user_kwargs: Dict) -> Dict:
        """Merge default parameters with user-provided parameters."""
        defaults = self.DEFAULT_PARAMS.get(self.scheduler_type, {}).copy()
        defaults.update(user_kwargs)
        return defaults

    def _validate_parameters(self):
        """Validate scheduler parameters."""
        if self.warmup_type not in ["linear", "exponential"]:
            raise ValueError(f"Invalid warmup type: {self.warmup_type}")
        if self.scheduler_type not in self.DEFAULT_PARAMS:
            raise ValueError(f"Unsupported scheduler type: {self.scheduler_type}")

    def _create_main_scheduler(self) -> _LRScheduler:
        """Create the main scheduler after warmup."""
        remaining_steps = self.total_steps - self.warmup_steps
        
        if self.scheduler_type == "cosine":
            return CosineAnnealingLR(
                self.optimizer, 
                T_max=remaining_steps,
                eta_min=self.params["eta_min"]
            )
        
        elif self.scheduler_type == "cosine_restart":
            T_0 = max(1, remaining_steps // self.params["rounds"])
            return CosineAnnealingWarmRestarts(
                self.optimizer,
                T_0=T_0,
                T_mult=self.params["T_mult"],
                eta_min=self.params["eta_min"]
            )
        
        elif self.scheduler_type == "step":
            step_size = max(1, remaining_steps // self.params["decay_steps"])
            return StepLR(
                self.optimizer,
                step_size=step_size,
                gamma=self.params["gamma"]
            )
        
        elif self.scheduler_type == "exponential":
            return ExponentialLR(
                self.optimizer,
                gamma=self.params["gamma"]
            )
        
        elif self.scheduler_type == "one_cycle":
            # Get base learning rates safely
            base_lrs = getattr(self, 'base_lrs', [group['lr'] for group in self.optimizer.param_groups])
            return OneCycleLR(
                self.optimizer,
                max_lr=[base_lr * self.params["lr_mult"] for base_lr in base_lrs],
                total_steps=self.total_steps,
                pct_start=self.warmup_ratio,
                anneal_strategy='cos',
                div_factor=self.params["div_factor"],
                final_div_factor=self.params["final_div_factor"]
            )

    def get_lr(self):
        """Get current learning rate."""
        if self._is_warmup:
            progress = min(1.0, self.current_step / self.warmup_steps)
            if self.warmup_type == "linear":
                factor = progress
            else:  # exponential
                factor = math.exp(progress * math.log(100)) / 100
            return [base_lr * factor for base_lr in self.base_lrs]
        
        # Return base learning rates if after_scheduler is not yet created
        if self.after_scheduler is None:
            return self.base_lrs
        
        return self.after_scheduler.get_last_lr()

    def step(self):
        """Step the scheduler."""
        self.current_step += 1
        if self._is_warmup and self.current_step >= self.warmup_steps:
            self._is_warmup = False
        
        if self._is_warmup:
            super().step()
        else:
            # Only step the after_scheduler if it's been created
            if self.after_scheduler is not None:
                self.after_scheduler.step()


def get_scheduler(args, optimizer, num_samples):
    """
    Create a learning rate scheduler from training arguments.
    
    Args:
        args: Training arguments object containing scheduler configuration
              Expected attributes:
              - batch_size: Training batch size
              - num_epoch: Number of training epochs
              - scheduler_type: Type of scheduler (default: 'cosine_restart')
              - warmup_ratio: Warmup ratio (default: 0.1)
              - warmup_type: Warmup type (default: 'linear')
              - eta_min, T_mult, rounds, gamma, decay_steps: Optional scheduler-specific params
        optimizer: PyTorch optimizer
        num_samples: Number of training samples
        
    Returns:
        AutoScheduler instance
        
    Example:
        # In any trainer class:
        self.optimizer = optim.AdamW(model.parameters(), lr=args.lr)
        self.scheduler = get_scheduler(args, self.optimizer, len(dataset))
        
        # During training:
        self.scheduler.step()
    """
    # Extract scheduler-specific parameters from args
    scheduler_kwargs = {}
    for param in ['eta_min', 'T_mult', 'rounds', 'gamma', 'decay_steps']:
        if hasattr(args, param):
            scheduler_kwargs[param] = getattr(args, param)
    
    # Calculate total steps and create scheduler
    total_steps = math.ceil(num_samples / args.batch_size) * args.num_epoch
    
    return AutoScheduler(
        optimizer=optimizer,
        total_steps=total_steps,
        scheduler_type=args.scheduler_type,
        warmup_ratio=getattr(args, 'warmup_ratio', 0.1),
        warmup_type=getattr(args, 'warmup_type', 'linear'),
        **scheduler_kwargs
    )