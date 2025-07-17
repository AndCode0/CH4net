import torch
import torch.nn as nn
import time

def create_loss_function(
    loss_type, focal_alpha=0.25, focal_gamma=2.0, pos_weight=1.0, smooth=1.0
):
    if loss_type == "bce":
        pos_weight_tensor = None
        if pos_weight != 1.0:
            pos_weight_tensor = torch.tensor([pos_weight])
        else:
            pos_weight_tensor = None

        return BCELogitsLoss(pos_weight=pos_weight_tensor)

    elif loss_type == "focal":
        return FocalLoss(alpha=focal_alpha, gamma=focal_gamma)

    elif loss_type == "dice":
        return DiceLoss(smooth=smooth)

    elif loss_type == "combined":
        return CombinedLoss(bce_weight=pos_weight, dice_weight=smooth, smooth=smooth)

    else:
        raise ValueError(f"Unknown loss function: {loss_type}")
    
    
class BCELogitsLoss(nn.Module):
    def __init__(self, pos_weight=None):
        super().__init__()
        self.bce_loss = nn.BCEWithLogitsLoss(reduction="none", pos_weight=pos_weight)

    def forward(self, pred, target):
        ll = self.bce_loss(pred, target)
        ll = ll.sum(dim=(-2, -1))  # Sum over (H, W)
        return ll.mean()  # Mean over batch dimension only


class FocalLoss(nn.Module):

    def __init__(self, alpha=0.25, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, pred, target):
        probs = torch.sigmoid(pred)
        bce = nn.BCEWithLogitsLoss(reduction="none")
        bce_loss = bce(pred, target)

        pt = probs * target + (1 - probs) * (1 - target)
        focal_weight = (1 - pt) ** self.gamma

        alpha_t = self.alpha * target + (1 - self.alpha) * (1 - target)
        bce_loss = alpha_t * bce_loss

        loss = focal_weight * bce_loss

        return loss.sum(dim=(-2, -1)).mean()


class DiceLoss(nn.Module):

    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        pred_probs = torch.sigmoid(pred)

        pred_flat = pred_probs.view(pred_probs.size(0), -1)  # (B, H*W)
        target_flat = target.view(target.size(0), -1)  # (B, H*W)

        intersection = (pred_flat * target_flat).sum(dim=1)  # Sum over spatial
        union = pred_flat.sum(dim=1) + target_flat.sum(dim=1)

        dice_coeff = (2.0 * intersection + self.smooth) / (union + self.smooth)
        dice_loss = 1 - dice_coeff

        return dice_loss.mean()  # Average over batch


class CombinedLoss(nn.Module):
    """Combined BCE + Dice Loss"""

    def __init__(self, bce_weight=1.0, dice_weight=1.0, smooth=1e-6):
        super(CombinedLoss, self).__init__()
        self.bce_loss = nn.BCEWithLogitsLoss(reduction="none")
        self.dice_loss = DiceLoss(smooth=smooth)
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight

    def forward(self, pred, target):
        # BCE loss
        bce = self.bce_loss(pred, target)
        bce = bce.sum(dim=(-2, -1)).mean()

        # Dice loss
        dice = self.dice_loss(pred, target)

        return self.bce_weight * bce + self.dice_weight * dice
    
class EarlyStopping:
    
    def __init__(self, config, save_path):
        self.enabled = config.get('enabled', False)
        if not self.enabled:
            return
            
        self.patience = config.get('patience', 20)
        self.min_delta = config.get('min_delta', 1e-4)
        self.monitor = config.get('monitor', 'val_loss')
        self.mode = config.get('mode', 'min')
        self.restore_best_weights = config.get('restore_best_weights', True)
        self.target_loss = config.get('target_loss', None)
        self.max_training_time = config.get('max_training_time_hours', None)
        self.divergence_threshold = config.get('divergence_threshold', None)
        self.min_epochs = config.get('min_epochs', 10)
        
        self.save_path = save_path
        self.best_score = float('inf') if self.mode == 'min' else float('-inf')
        self.best_epoch = 0
        self.wait = 0
        self.stopped_epoch = 0
        self.start_time = time.time()
        
        print(f"Early stopping enabled: monitor={self.monitor}, patience={self.patience}, mode={self.mode}")
    
    def __call__(self, epoch, current_score, model):
        if not self.enabled:
            return False
        
        # Check minimum epochs
        if epoch < self.min_epochs:
            return False
        
        # Check target loss
        if self.target_loss is not None and current_score <= self.target_loss:
            print(f"\t\tTarget loss {self.target_loss} reached at epoch {epoch}")
            self.stopped_epoch = epoch
            self.stop_reason = f"Target loss {self.target_loss} reached"
            return True
        
        # Check maximum training time
        if self.max_training_time is not None:
            elapsed_hours = (time.time() - self.start_time) / 3600
            if elapsed_hours >= self.max_training_time:
                print(f"⏰ Maximum training time ({self.max_training_time}h) reached")
                self.stopped_epoch = epoch
                self.stop_reason = f"Maximum training time ({self.max_training_time}h) reached"
                return True
        
        # Check divergence
        if self.divergence_threshold is not None and current_score > self.divergence_threshold:
            print(f"Loss diverged above threshold {self.divergence_threshold}")
            self.stopped_epoch = epoch
            self.stop_reason = f"Loss diverged above threshold {self.divergence_threshold}"
            return True
        
        # Check improvement
        if self.mode == 'min':
            improved = current_score < (self.best_score - self.min_delta)
        else:
            improved = current_score > (self.best_score + self.min_delta)
        
        if improved:
            self.best_score = current_score
            self.best_epoch = epoch
            self.wait = 0
        
        else:
            self.wait += 1
            
        # Check patience
        if self.wait >= self.patience:
            self.stopped_epoch = epoch
            self.stop_reason = f"No improvement for {self.wait} epochs"
            print(f"Early stopping triggered after {self.wait} epochs without improvement")
            print(f"\tBest {self.monitor}: {self.best_score:.6f} at epoch {self.best_epoch}")
            
            return True
        
        return False
    
    def get_summary(self):
        if not self.enabled:
            return "Early stopping was disabled"
        
        if self.stopped_epoch > 0:
            return f"Early stopping at epoch {self.stopped_epoch}: {self.stop_reason}. Best {self.monitor}: {self.best_score:.6f} at epoch {self.best_epoch}"
        else:
            return f"Training completed without early stopping. Best {self.monitor}: {self.best_score:.6f} at epoch {self.best_epoch}"