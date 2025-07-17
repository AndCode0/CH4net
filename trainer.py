import gc
import os
import torch
import optuna
import logging
import torchvision
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from utils import SeededWorkerInitializer
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau


class Trainer:
    """
    Training class 
    """

    def __init__(
        self,
        model,
        train_dataset,
        val_loader,
        loss_function,
        save_path,
        device,
        seed,
        learning_rate,
        multimetrics,
        optimizer=None,
        scheduler=None,
        dataloader_config={},
        training_config={},
        tensorboard_config={},
        generator=None,
        sweep_mode=False,
        trial=None,
    ):

        # Model and data
        self.model = model
        self.train_dataset = train_dataset
        self.val_loader = val_loader
        self.save_path = save_path
        self.device = device
        self.scheduler = scheduler
        self.sweep_mode = sweep_mode
        self.trial = trial
        self.multimetrics = multimetrics

        # Training parameters
        self.batch_size = training_config["batch_size"]
        self.save_every_n_epochs = training_config["save_every_n_epochs"]
        self.cleanup_every_n_batches = training_config["cleanup_every_n_batches"]
        self.scaler = torch.GradScaler(enabled=True)

        # Tensorboard parameters
        if not sweep_mode:
            self.log_images_every_n_epochs = tensorboard_config["log_images_every_n_epochs"]
            
            self.log_histograms_every_n_epochs = tensorboard_config["log_histograms_every_n_epochs"]
            
            self.log_batch_every_n_batches = tensorboard_config["log_batch_every_n_batches"]
            
            self.max_images_to_log = tensorboard_config["max_images_to_log"]
            self.normalize_images = tensorboard_config["normalize_images"]
            self.nrow_images = tensorboard_config["nrow_images"]

            try:
                self.log_dir = os.path.join(save_path, "tensorboard_logs")
                os.makedirs(self.log_dir, exist_ok=True)
                self.writer = SummaryWriter(self.log_dir)
            except Exception as e:
                logging.warning(f"Could not initialize TensorBoard writer: {e}")
                self.writer = None
        else:
            self.writer = None

        # Dataloader parameters
        num_workers = dataloader_config["num_workers"]
        pin_memory = dataloader_config["pin_memory"]
        persistent_workers = dataloader_config["persistent_workers"]
        prefetch_factor = dataloader_config["prefetch_factor"]

        worker_init_fn = SeededWorkerInitializer(seed)

        # Create train_loader ONCE with custom sampler
        train_sampler = train_dataset.get_sampler()
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            sampler=train_sampler,
            pin_memory=pin_memory,
            num_workers=num_workers,
            persistent_workers=persistent_workers if num_workers > 0 else False,
            prefetch_factor=prefetch_factor if num_workers > 0 else 2,
            generator=generator,
            worker_init_fn=worker_init_fn,
            # Note: Can't use shuffle=True when using a custom sampler
        )

        # Training parameters
        if optimizer is not None:
            self.opt = optimizer
        else:
            # Default to Adam if no optimizer provided
            self.opt = torch.optim.Adam(model.parameters(), lr=learning_rate)
        self.loss_function = loss_function

        # Losses and metrics
        self.losses = []
        self.best_loss = float("inf")
        self.best_f1 = 0.0
        self.best_iou = 0.0

        # Resource tracking
        self._resources_initialized = True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup_resources()

    def cleanup_resources(self):
        try:
            if hasattr(self, "_resources_initialized") and self._resources_initialized:
                # Close TensorBoard writer
                if hasattr(self, "writer") and self.writer is not None:
                    self.writer.close()
                    self.writer = None

                # Clear CUDA cache
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()

                # Force garbage collection
                gc.collect()

                self._resources_initialized = False

        except Exception as e:
            logging.warning(f"Error during resource cleanup: {e}")

    def close(self):
        self.cleanup_resources()

    def compute_metrics(self, predictions, targets, threshold=0.5):

        with torch.no_grad():
            # Convert logits to probabilities
            pred_probs = torch.sigmoid(predictions)
            pred_binary = (pred_probs > threshold).float()

            # Flatten tensors
            pred_flat = pred_binary.view(-1)
            target_flat = targets.view(-1)

            # Compute confusion matrix components
            tp = (pred_flat * target_flat).sum().item()
            fp = (pred_flat * (1 - target_flat)).sum().item()
            fn = ((1 - pred_flat) * target_flat).sum().item()
            tn = ((1 - pred_flat) * (1 - target_flat)).sum().item()

            return {
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "tn": tn,
            }

    def eval_epoch(self, epoch=None, verbose=False, compute_detailed_metrics=False):
        self.model.eval()

        running_loss = 0.0
        num_batches = 0

        total_tp = total_fp = total_fn = 0

        if verbose:
            outs = []
            ts = []

        with torch.no_grad():
            for batch_idx, task in enumerate(self.val_loader):
                pred = task["pred"].to(self.device, non_blocking=True)
                target = task["target"].to(self.device, non_blocking=True)

                out = self.model(pred)
                logits = out[..., 0]
                loss_val = self.loss_function(logits, target)

                running_loss += loss_val.item()
                num_batches += 1

                if compute_detailed_metrics or self.sweep_mode:
                    # Compute metrics for this batch
                    batch_metrics = self.compute_metrics(logits, target)
                    total_tp += batch_metrics["tp"]
                    total_fp += batch_metrics["fp"]
                    total_fn += batch_metrics["fn"]
                    total_tn += batch_metrics["tn"]

                if verbose:
                    outs.append(out.detach().cpu().numpy())
                    ts.append(target.detach().cpu().numpy())

                # Periodic cleanup
                if (
                    batch_idx % self.cleanup_every_n_batches == 0
                    and torch.cuda.is_available()
                ):
                    torch.cuda.empty_cache()

        if num_batches == 0:
            logging.error("No validation batches processed successfully")
            raise ValueError("0 batches")

        # Calculate average loss
        avg_loss = running_loss / num_batches

        metrics = {"loss": avg_loss}
        if compute_detailed_metrics or self.sweep_mode:
            epsilon = 1e-8
            precision = total_tp / (total_tp + total_fp + epsilon)
            recall = total_tp / (total_tp + total_fn + epsilon)
            f1 = 2 * precision * recall / (precision + recall + epsilon)

            intersection = total_tp
            union = total_tp + total_fp + total_fn
            iou = intersection / (union + epsilon)

            metrics.update(
                {
                    "f1": f1,
                    "precision": precision,
                    "recall": recall,
                    "iou": iou,
                    "tp": total_tp,
                    "fp": total_fp,
                    "fn": total_fn,
                    "tn": total_tn,
                }
            )

        print(f"- Validation loss: {avg_loss:.6f}")
        if "f1" in metrics:
            print(f"- F1: {metrics['f1']:.4f}, IoU: {metrics['iou']:.4f}")

        if epoch is not None and self.writer is not None:
            try:
                self.writer.add_scalar("Loss/Validation", avg_loss, epoch)
                if "f1" in metrics:
                    self.writer.add_scalar("Metrics/F1", metrics["f1"], epoch)
                    self.writer.add_scalar("Metrics/IoU", metrics["iou"], epoch)
                    self.writer.add_scalar(
                        "Metrics/Precision", metrics["precision"], epoch
                    )
                    self.writer.add_scalar("Metrics/Recall", metrics["recall"], epoch)
            except Exception as e:
                logging.warning(f"Failed to log validation loss: {e}")

        if verbose:
            if outs and ts:
                return metrics, np.concatenate(outs, axis=0), np.concatenate(ts, axis=0)
            else:
                raise ValueError(f"\n\touts: {outs}, \n\tts: {ts}")
        else:
            return metrics, None, None

    def report_to_optuna(self, epoch, metrics, objective_metric="f1"):
        if self.trial is None:
            return False

        try:
            objective_value = metrics[objective_metric]

            # For F1 and IoU, we want to maximize, so report negative value
            if objective_metric in ["f1", "iou"]:
                report_value = -objective_value
            else:  # For loss, we want to minimize
                report_value = objective_value

            self.trial.report(report_value, epoch)

            if self.trial.should_prune():
                logging.info(f"Trial {self.trial.number} pruned at epoch {epoch}")
                return True

        except Exception as e:
            logging.warning(f"Error reporting to Optuna: {e}")

        return False

    def train(
        self,
        objective_metric,
        n_epochs=100,
        start_epoch=0,
        early_stopping=None,
        report_interval=5,
    ):

        for epoch in range(start_epoch, start_epoch + n_epochs):
            # Uncomment for debugging
            # autograd.set_detect_anomaly(True)
            print(f"Training epoch {epoch}")
            self.model.train()

            running_train_loss = 0.0
            num_train_batches = 0

            with tqdm(
                self.train_loader, unit="batch", disable=self.sweep_mode
            ) as tepoch:
                for batch_idx, task in enumerate(tepoch):
                    pred = task["pred"].to(self.device, non_blocking=True)
                    target = task["target"].to(self.device, non_blocking=True)

                    with torch.autocast(device_type=self.device.type, enabled=True):
                        out = self.model(pred)
                        logits = out[..., 0]
                        loss = self.loss_function(logits, target)

                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.opt)
                    self.scaler.update()
                    self.opt.zero_grad()

                    running_train_loss += loss.item()
                    num_train_batches += 1
                    if not self.sweep_mode:
                        tepoch.set_postfix(loss=loss.item())

                    # Log training loss every N batches
                    if (
                        self.writer is not None
                        and not self.sweep_mode
                        and batch_idx % self.log_batch_every_n_batches == 0
                    ):
                        global_step = epoch * len(self.train_loader) + batch_idx
                        self.writer.add_scalar(
                            "Loss/Training_Batch", loss.item(), global_step
                        )

                    # Periodic cleanup
                    if (
                        batch_idx % self.cleanup_every_n_batches == 0
                        and torch.cuda.is_available()
                    ):
                        torch.cuda.empty_cache()

            avg_train_loss = running_train_loss / num_train_batches
            if self.writer is not None:
                self.writer.add_scalar("Loss/Training_Epoch", avg_train_loss, epoch)

            # Validation
            metrics, *_ = self.eval_epoch(
                epoch=epoch, verbose=False, compute_detailed_metrics=self.multimetrics
            )
            epoch_loss = metrics["loss"]

            # Update best metrics
            if epoch_loss < self.best_loss:
                try:
                    self.best_loss = epoch_loss
                    save_path = os.path.join(self.save_path, f"best_model.pt")
                    checkpoint_data = {
                        "epoch": epoch,
                        "model_state_dict": self.model.state_dict(),
                        "optimizer_state_dict": self.opt.state_dict(),
                        "loss": epoch_loss,
                        "best_loss": self.best_loss,
                        "optimizer_type": type(self.opt).__name__,
                        "metrics": metrics,
                    }

                    torch.save(checkpoint_data, save_path)
                    print(
                        f"Saved best model at epoch {epoch} with loss {epoch_loss:.6f}"
                    )

                    if self.trial is not None:
                        checkpoint_data["trial_params"] = self.trial.params
                        checkpoint_data["trial_number"] = self.trial.number

                    # Log best loss
                    if self.writer is not None and not self.sweep_mode:
                        self.writer.add_scalar(
                            "Loss/Best_Validation", self.best_loss, epoch
                        )
                        self.writer.add_scalar("Metrics/Best_F1", self.best_f1, epoch)
                        self.writer.add_scalar("Metrics/Best_IoU", self.best_iou, epoch)

                except Exception as e:
                    logging.error(f"Failed to save checkpoint: {e}")

            if metrics.get("f1", 0) > self.best_f1:
                self.best_f1 = metrics["f1"]
            if metrics.get("iou", 0) > self.best_iou:
                self.best_iou = metrics["iou"]

            # Check early stopping (only if not in sweep mode)
            if early_stopping is not None and not self.sweep_mode:
                monitor_value = metrics.get(objective_metric, epoch_loss) if objective_metric != 'loss' else epoch_loss
                if early_stopping(epoch, monitor_value, self.model):
                    print(f" Early stopping triggered at epoch {epoch}")
                    break

            # Report to Optuna and check pruning (only in sweep mode)
            if self.sweep_mode and epoch % report_interval == 0:
                should_prune = self.report_to_optuna(epoch, metrics, objective_metric)
                if should_prune:
                    raise optuna.TrialPruned()

            # Learning rate scheduling
            if self.scheduler is not None:
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    self.scheduler.step(epoch_loss)
                else:
                    self.scheduler.step()

                # Log learning rate
                if self.writer is not None:
                    current_lr = self.opt.param_groups[0]["lr"]
                    self.writer.add_scalar("Learning_Rate", current_lr, epoch)        

            # Save periodic checkpoints
            if epoch % self.save_every_n_epochs == 0 and not self.sweep_mode:
                try:
                    checkpoint_path = os.path.join(
                        self.save_path, f"checkpoint_epoch_{epoch}.pt"
                    )
                    torch.save(
                        {
                            "epoch": epoch,
                            "model_state_dict": self.model.state_dict(),
                            "optimizer_state_dict": self.opt.state_dict(),
                            "loss": epoch_loss,
                            "best_loss": self.best_loss,
                            "optimizer_type": type(self.opt).__name__,
                            "metrics": metrics,
                        },
                        checkpoint_path,
                    )
                    print(f"Saved checkpoint at epoch {epoch}")
                except Exception as e:
                    logging.warning(f"Failed to save checkpoint: {e}")

            # Log sample images every 5 epochs
            if (
                epoch % self.log_images_every_n_epochs == 0
                and self.writer is not None
                and not self.sweep_mode
            ):
                try:
                    metrics_detailed, predictions, targets = self.eval_epoch(
                        epoch=epoch, verbose=True
                    )
                    if predictions is not None and targets is not None:
                        self._log_images_to_tensorboard(epoch, predictions, targets)
                except Exception as e:
                    logging.warning(f"Failed to log images to TensorBoard: {e}")

            # Log model parameters every 10 epochs
            if epoch % self.log_histograms_every_n_epochs == 0 and not self.sweep_mode:
                try:
                    self._log_histograms(epoch)
                except Exception as e:
                    logging.warning(f"Failed to log histograms: {e}")

        if not self.sweep_mode:
            print("Training complete!")
        else:
            final_metrics = self.eval_epoch(verbose=False, compute_detailed_metrics=True)
            return final_metrics

        # Final cleanup
        try:
            self.cleanup_resources()
        except:
            pass

    def _log_images_to_tensorboard(self, epoch, predictions, targets, max_images=8):
        """
        Log sample predictions and targets to TensorBoard
        """
        # Limit number of images to log
        n_images = min(max_images, self.batch_size)

        pred_imgs = predictions[:n_images]  # Shape: (N, H, W, 1) or (N, H, W)
        target_imgs = targets[:n_images]  # Shape: (N, H, W)

        # Ensure correct shape for tensorboard (add channel dimension if needed)
        if len(pred_imgs.shape) == 4 and pred_imgs.shape[-1] == 1:
            pred_imgs = pred_imgs.squeeze(-1)  # Remove last dimension if it's 1
        if len(target_imgs.shape) == 3:
            target_imgs = target_imgs  # Keep as is

        # Convert to tensors and add batch dimension for tensorboard
        pred_tensor = torch.from_numpy(pred_imgs).unsqueeze(1)  # Add channel dim
        target_tensor = torch.from_numpy(target_imgs).unsqueeze(1)  # Add channel dim

        # Log predictions and targets
        img_grid_pred = torchvision.utils.make_grid(
            pred_tensor, normalize=self.normalize_images, nrow=4
        )
        img_grid_target = torchvision.utils.make_grid(
            target_tensor, normalize=self.normalize_images, nrow=4
        )

        # Log difference/error maps
        diff = np.abs(pred_imgs - target_imgs)
        diff_tensor = torch.from_numpy(diff).unsqueeze(1)
        img_grid_diff = torchvision.utils.make_grid(diff_tensor, normalize=True, nrow=4)
        if self.writer:
            self.writer.add_image("Predictions", img_grid_pred, epoch)
            self.writer.add_image("Targets", img_grid_target, epoch)
            self.writer.add_image("Prediction_Error", img_grid_diff, epoch)

    def _log_histograms(self, epoch):
        """
        Log model parameter histograms
        """
        if self.writer:
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    self.writer.add_histogram(name, param, epoch)
                    if param.grad is not None:
                        self.writer.add_histogram(f"{name}_grad", param.grad, epoch)
