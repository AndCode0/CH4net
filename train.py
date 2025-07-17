from models import Unet
from utils import (
    merge_configs,
    save_complete_config,
    set_all_seeds,
    SeededWorkerInitializer,
    create_optimizer,
    create_scheduler,
    config_for_Tester,
    load_checkpoint,
)
from trainer import Trainer
from loader import MethaneLoader
from losses import create_loss_function, EarlyStopping
import os
import argparse
from torch.utils.data import DataLoader
from torch.nn import DataParallel
import torch


def parse_arguments():
    parser = argparse.ArgumentParser(
        description=("Command-line argument parser for training script `train.py`. "),
        epilog=(
            "Example usage:\n"
            "  python train.py -c ./configs/training_config.json --outdir ./results\n"
            "Note: All required arguments must be provided. For more help, use -h or --help."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Required arguments
    parser.add_argument(
        "-d",
        "--outdir",
        required=True,
        help="Path to the directory where output files, logs, and models will be saved.",
    )

    # Sweep parameters
    parser.add_argument(
        "-c",
        "--config_path",
        type=str,
        default=None,
        help="Path to training_config.json file",
    )

    parser.add_argument(
        "--compute-metrics",
        action="store_true",
        help="Compute and log F1/IoU metrics during training",
    )

    return parser.parse_args()


def main():
    args = parse_arguments()

    # Load config from .json and merge it with the output from the params sweep
    try:
        config = merge_configs(args.config_path)
    except Exception as e:
        print(f" Error loading configuration: {e}")
        return

    metrics = args.compute_metrics
    args.compute_metrics = True if (config["metric"] != "loss") else metrics

    # Create output directory if it doesn't exist
    os.makedirs(args.outdir, exist_ok=True)

    # Save final configuration
    config_path = os.path.join(args.outdir, "final_config.json")
    save_complete_config(config, config_path, format_type="nested")
    print(f"Saved training configuration to: {config_path}")

    # Set up device and random seed
    set_all_seeds(config["random_seed"])

    if config["cuda_benchmark"]:
        torch.backends.cudnn.benchmark = True
    device = (
        torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    )
    if config["device"] != "auto":
        device = torch.device(config["device"])
    
    print(f"\n\tUsing device: {device}")

    model = Unet(
        in_channels=config["channels"],
        out_channels=config["out_channels"],
        div_factor=config["div_factor"],
        prob_output=config["prob_output"],
    )
    model = model.to(device)

    if torch.cuda.device_count() > 1:
        print(f"\tUsing {torch.cuda.device_count()} GPUs")
        model = DataParallel(model)

    # Set up optimizer
    optimizer = create_optimizer(
        model,
        config["optimizer"],
        config["learning_rate"],
        config["momentum"],
        config["weight_decay"],
    )
    print(f"\tUsing optimizer: {config['optimizer']}")

    # Set up loss function
    loss_function = create_loss_function(
        config["loss_type"],
        focal_alpha=config["focal_alpha"],
        focal_gamma=config["focal_gamma"],
        pos_weight=config["pos_weight"],
        smooth=config["smooth"],
    )
    loss_function = loss_function.to(device)
    print(f"\tUsing loss function: {config["loss_type"]}")

    # Set up scheduler
    scheduler = None
    if config.get("use_scheduler", False):
        step_size = config["epochs"] // 3
        t_ = config.get("t_max", None)
        t_max = t_ if t_ is not None else config["epochs"]
        eta_min = config["learning_rate"] / 10
        scheduler = create_scheduler(
            optimizer,
            config["scheduler_type"],
            patience=config["plateau_patience"],
            factor=config["plateau_factor"],
            step_size=step_size,
            gamma=config["gamma"],
            t_max=t_max,
            eta_min=eta_min,
        )
        print(f"\tUsing scheduler: {config['scheduler_type']}")
    else:
        print("\t--No scheduler configured")

    # Load checkpoint if provided
    start_epoch = config.get("resume_epoch", 0)
    best_loss = float("inf")
    if config.get("from_sweep", False):
        best_loss = config.get("best_value", float("inf"))
        print(f"Using best loss from sweep: {best_loss:.6f}")

    checkpoint_path = config.get("checkpoint")
    if checkpoint_path:
        start_epoch, best_loss = load_checkpoint(
            checkpoint_path, model, optimizer, device
        )
        start_epoch += 1
        print(f"Resuming training from epoch {start_epoch}")
    elif start_epoch == 0:
        print("Starting training from scratch")

    train_dataset = MethaneLoader(
        mode="train",
        channels=config["channels"],
        base_seed=config["random_seed"],
        neg_ratio=config.get("neg_ratio", 1.0),
        generator=torch.Generator().manual_seed(config["random_seed"]),
        path=config["base_path"],
    )
    test_dataset = MethaneLoader(
        mode="val",
        channels=config["channels"],
        base_seed=config["random_seed"],
        neg_ratio=config.get("neg_ratio", 1.0),
        generator=torch.Generator().manual_seed(config["random_seed"] + 1),
        path=config["base_path"],
    )

    worker_init_fn = SeededWorkerInitializer(config["random_seed"])

    test_loader = DataLoader(
        test_dataset,
        batch_size=config["batch_size"],
        sampler=test_dataset.get_sampler(),
        pin_memory=config["pin_memory"],
        num_workers=config["num_workers"],
        persistent_workers=config["persistent_workers"],
        prefetch_factor=config["prefetch_factor"],
        generator=torch.Generator().manual_seed(config["random_seed"] + 2),
        worker_init_fn=worker_init_fn,
    )

    trainer_config = config_for_Tester(config)
    sweep_mode = config.get("sweep_mode", False)
    disable_tensorboard = config.get("disable_tensorboard", False) or sweep_mode
    disable_early_stopping = config.get("disable_early_stopping", False) or sweep_mode

    # Set up trainer
    trainer = Trainer(
        model=model,
        train_dataset=train_dataset,
        val_loader=test_loader,
        loss_function=loss_function,
        save_path=args.outdir,
        device=device,
        multimetrics=args.compute_metrics,
        optimizer=optimizer,
        scheduler=scheduler,
        learning_rate=config["learning_rate"],
        dataloader_config=trainer_config["data"],
        training_config=trainer_config["training"],
        tensorboard_config=(
            trainer_config["tensorboard"] if not disable_tensorboard else {}
        ),
        generator=torch.Generator().manual_seed(config["random_seed"] + 3),
        seed=config["random_seed"],
    )

    # Set the best loss if we loaded from checkpoint
    trainer.best_loss = best_loss

    early_stopping = None
    if not disable_early_stopping:
        early_stopping_config = config.get("early_stopping", {})
        if early_stopping_config.get("enabled", False):
            early_stopping = EarlyStopping(early_stopping_config, args.outdir)
            print(" \tEarly stopping enabled")
        else:
            print("\t--Early stopping disabled in configuration")
    else:
        print("\t--Early stopping disabled (sweep mode or explicitly disabled)")

    # Print TensorBoard info
    print(f"To view TensorBoard during training, run:")
    print(f"tensorboard --logdir {trainer.log_dir}")

    # Calculate remaining epochs
    remaining_epochs = config["epochs"] - start_epoch
    if remaining_epochs <= 0:
        print(f"Model has already been trained for {config['epochs']} epochs!")
        return

    print(
        f"Training for {remaining_epochs} epochs (from epoch {start_epoch} to {config['epochs']})"
    )

    # Train the model
    try:
        if args.compute_metrics or sweep_mode:
            # Use enhanced training with metrics
            trainer.train(
                n_epochs=remaining_epochs,
                start_epoch=start_epoch,
                early_stopping=early_stopping,
                objective_metric=config["metric"],  # TODO: name for metrics computation
                report_interval=5,
            )
        else:
            trainer.train(
                n_epochs=remaining_epochs,
                start_epoch=start_epoch,
                early_stopping=early_stopping,
                objective_metric="loss",
            )
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    except Exception as e:
        print(f"Training failed with error: {e}")
        raise
    finally:
        # Clean up resources
        trainer.cleanup_resources()

    print("Training completed!")
    print(f"\nFinal metrics:")
    print(f"  Best Loss: {trainer.best_loss:.6f}")
    if args.compute_metrics:
        print(f"  Best F1: {trainer.best_f1:.4f}")
        print(f"  Best IoU: {trainer.best_iou:.4f}")


if __name__ == "__main__":
    main()
