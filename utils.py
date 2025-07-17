import os
import json
import torch
import numpy as np
import logging
import random
from typing import Dict
from torch.optim import Adam, SGD, AdamW
from torch.optim.lr_scheduler import (
    ReduceLROnPlateau,
    CosineAnnealingLR,
    StepLR,
    LinearLR,
    SequentialLR,
)


# =====================
#      CONFIGURATION
# ======================
def load_params(params_path):

    if not os.path.exists(params_path):
        raise FileNotFoundError(f"Best params file not found: {params_path}")

    with open(params_path, "r") as f:
        params = json.load(f)

    return params


def unflatten_config(flat_config):
    # Define which parameters belong to which sections
    section_mapping = {
        "model": ["channels", "div_factor", "prob_output", "out_channels"],
        "training": [
            "epochs",
            "batch_size",
            "resume_epoch",
            "save_every_n_epochs",
            "cleanup_every_n_batches",
            "metric",
        ],
        "optimizer": ["optimizer", "learning_rate", "momentum", "weight_decay"],
        "loss_function": [
            "loss_type",
            "focal_alpha",
            "focal_gamma",
            "combo_alpha",
            "combo_beta",
            "dice_smooth",
            "pos_weight",
            "smooth",
        ],
        "scheduler": [
            "use_scheduler",
            "scheduler_type",
            "plateau_patience",
            "plateau_factor",
            "threshold",
            "min_lr",
            "t_max",
            "step_size",
            "gamma",
        ],
        "stopping": ["early_stopping"],
        "data": [
            "num_workers",
            "pin_memory",
            "persistent_workers",
            "prefetch_factor",
            "neg_ratio",
            "base_path",
        ],
        "system": ["device", "cuda_benchmark", "random_seed"],
        "tensorboard": [
            "log_dir_name",
            "max_images_to_log",
            "normalize_images",
            "nrow_images",
            "log_images_every_n_epochs",
            "log_histograms_every_n_epochs",
            "log_batch_every_n_batches",
        ],
        "paths": ["checkpoint", "sweep_override"],
    }

    nested_config = {}

    # Initialize all sections
    for section in section_mapping:
        nested_config[section] = {}

    # Distribute parameters to appropriate sections
    for key, value in flat_config.items():
        placed = False
        for section, params in section_mapping.items():
            if key in params:
                nested_config[section][key] = value
                placed = True
                break

        # If parameter doesn't fit in any section, add to a misc section
        if not placed:
            if "misc" not in nested_config:
                nested_config["misc"] = {}
            nested_config["misc"][key] = value

    # Handle special nested structures
    if "early_stopping" in flat_config:
        nested_config["stopping"] = {"early_stopping": flat_config["early_stopping"]}

    return nested_config


def merge_configs(params_path) -> Dict:
    config_grouped = load_params(params_path)
    # Flatten dict
    config = {}
    for v in config_grouped.values():
        config.update(v)

    sweep_override_path = config.get("sweep_override")
    if sweep_override_path is not None:
        if os.path.exists(sweep_override_path):
            print("Merging sweep's best parameters into the training config ...")
            sweep = load_params(config["paths"]["sweep_override"])
            best_params = sweep.get("best_params", {})
        else:
            raise ValueError(f"the path to the sweep result doesn't exist")

        # Check if sweep['best_params'] is a subset of config
        invalid_keys = []
        for key in best_params:
            if key not in config and key not in [
                "sweep_mode",
                "disable_tensorboard",
                "disable_early_stopping",
            ]:
                invalid_keys.append(key)

        if invalid_keys:
            logging.warning(
                f"Sweep parameters not found in base config: {invalid_keys}"
            )
            # Add them anyway with a warning
            for key in invalid_keys:
                logging.warning(
                    f"Adding unknown parameter from sweep: {key} = {best_params[key]}"
                )

        # Update values
        config.update(sweep["best_params"])
        for meta_key in ["channels", "loss_type", "best_value", "objective_metric"]:
            if meta_key in sweep:
                config[meta_key] = sweep[meta_key]

        # Mark as coming from sweep
        config["from_sweep"] = True
        config["sweep_study_name"] = sweep.get("study_name", "unknown")
        config["sweep_n_trials"] = sweep.get("n_trials", 0)

    return config


def config_for_Tester(config):
    trainer_config = {}
    trainer_config["training"] = {
        "batch_size": config["batch_size"],
        "save_every_n_epochs": config["save_every_n_epochs"],
        "cleanup_every_n_batches": config["cleanup_every_n_batches"],
    }
    trainer_config["data"] = {
        "num_workers": config["num_workers"],
        "pin_memory": config["pin_memory"],
        "persistent_workers": config["persistent_workers"],
        "prefetch_factor": config["prefetch_factor"],
    }
    trainer_config["tensorboard"] = {
        "max_images_to_log": config["max_images_to_log"],
        "normalize_images": config["normalize_images"],
        "nrow_images": config["nrow_images"],
        "log_images_every_n_epochs": config["log_images_every_n_epochs"],
        "log_histograms_every_n_epochs": config["log_histograms_every_n_epochs"],
        "log_batch_every_n_batches": config["log_batch_every_n_batches"],
    }

    return trainer_config


def save_complete_config(config, output_path, format_type):
    if format_type == "nested":
        # Convert to nested format for compatibility
        nested_config = unflatten_config(config)
        save_config = nested_config
    else:
        # Save as flat dictionary
        save_config = config

    with open(output_path, "w") as f:
        json.dump(save_config, f, indent=2)


# ======================
#      MODEL SET UP
# ======================
def create_optimizer(
    model, optimizer_type, learning_rate, momentum=0.9, weight_decay=0.0
):

    if optimizer_type == "Adam":
        return Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    elif optimizer_type == "SGD":
        return SGD(
            model.parameters(),
            lr=learning_rate,
            momentum=momentum,
            weight_decay=weight_decay,
        )

    elif optimizer_type == "AdamW":
        return AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    else:
        raise ValueError(f"Unknown optimizer: {optimizer_type}")


def create_scheduler(
    optimizer,
    scheduler_type,
    t_max,
    patience=10,
    factor=0.5,
    gamma=0.1,
    step_size=30,
    eta_min=1e-5,
):

    if scheduler_type == "ReduceLROnPlateau":
        return ReduceLROnPlateau(
            optimizer, mode="min", patience=patience, factor=factor
        )
    elif scheduler_type == "CosineAnnealingLR":
        warmup_epochs = 5
        cosine_epochs = t_max - warmup_epochs
        scheduler_warmup = LinearLR(
            optimizer, start_factor=0.01, total_iters=warmup_epochs
        )
        scheduler_cos = CosineAnnealingLR(
            optimizer, T_max=cosine_epochs, eta_min=eta_min
        )
        return SequentialLR(
            optimizer,
            schedulers=[scheduler_warmup, scheduler_cos],
            milestones=[warmup_epochs],
        )

    elif scheduler_type == "StepLR":
        return StepLR(optimizer, step_size=step_size, gamma=gamma)
    else:
        raise ValueError(f"Unknown scheduler: {scheduler_type}")

def load_checkpoint(checkpoint_path, model, device, optimizer=None):
    print(f"Loading checkpoint from: {checkpoint_path}")

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Load model state
    model.load_state_dict(checkpoint["model_state_dict"])

    # Load optimizer state if provided
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    # Get epoch and loss info
    epoch = checkpoint.get("epoch", 0)
    loss_val = checkpoint.get("loss", float("inf"))

    print(f"Loaded checkpoint from epoch {epoch} with loss: {loss_val:.6f}")

    return epoch, loss_val

# ======================
#      SEED
# ======================

def set_all_seeds(seed: int):
    try:
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            # For reproducibility
            torch.backends.cudnn.deterministic = True
    except Exception as e:
        logging.warning(f"Error setting seeds: {e}")

class SeededWorkerInitializer:
    __slots__ = ('seed',)  # Makes it pickleable
    
    def __init__(self, seed):
        self.seed = seed
        
    def __call__(self, worker_id):
        worker_seed = self.seed + worker_id
        random.seed(worker_seed)
        np.random.seed(worker_seed)
        torch.manual_seed(worker_seed)

