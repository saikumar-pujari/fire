import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from tqdm import tqdm

class Trainer:
    """
    Production-grade training loop engine for Forest Fire Detection & Risk Classification.
    Supports Mixed Precision (AMP), Learning Rate Scheduling, Early Stopping, and Model Checkpointing.
    """
    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        criterion=None,
        optimizer=None,
        scheduler=None,
        device="cpu",
        use_amp=False,
        checkpoint_dir="outputs/checkpoints",
        model_name="resnet18"
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.use_amp = use_amp and (device != "cpu")
        self.checkpoint_dir = checkpoint_dir
        self.model_name = model_name

        os.makedirs(self.checkpoint_dir, exist_ok=True)

        self.criterion = criterion or nn.CrossEntropyLoss()
        self.optimizer = optimizer or optim.AdamW(self.model.parameters(), lr=1e-3, weight_decay=1e-4)
        self.scheduler = scheduler or CosineAnnealingLR(self.optimizer, T_max=10, eta_min=1e-6)
        
        # AMP Scaler
        if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
            self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)
        else:
            self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)

    def train_epoch(self):
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for imgs, targets in tqdm(self.train_loader, desc=f"Training [{self.model_name}]", leave=False):
            imgs = imgs.to(self.device)
            targets = targets.to(self.device)

            self.optimizer.zero_grad()

            if self.use_amp:
                with torch.cuda.amp.autocast():
                    outputs = self.model(imgs)
                    if targets.ndim > 1:
                        # Soft labels (MixUp/CutMix)
                        loss = torch.sum(-targets * torch.log_softmax(outputs, dim=1), dim=1).mean()
                    else:
                        loss = self.criterion(outputs, targets)

                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(imgs)
                if targets.ndim > 1:
                    loss = torch.sum(-targets * torch.log_softmax(outputs, dim=1), dim=1).mean()
                else:
                    loss = self.criterion(outputs, targets)
                
                loss.backward()
                self.optimizer.step()

            running_loss += loss.item() * imgs.size(0)
            
            if targets.ndim > 1:
                _, predicted = outputs.max(1)
                _, target_indices = targets.max(1)
                correct += predicted.eq(target_indices).sum().item()
            else:
                _, predicted = outputs.max(1)
                correct += predicted.eq(targets).sum().item()

            total += imgs.size(0)

        epoch_loss = running_loss / total
        epoch_acc = correct / total
        return epoch_loss, epoch_acc

    def validate(self):
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for imgs, targets in tqdm(self.val_loader, desc=f"Validation [{self.model_name}]", leave=False):
                imgs = imgs.to(self.device)
                targets = targets.to(self.device)

                outputs = self.model(imgs)
                if targets.ndim > 1:
                    loss = torch.sum(-targets * torch.log_softmax(outputs, dim=1), dim=1).mean()
                    _, predicted = outputs.max(1)
                    _, target_indices = targets.max(1)
                    correct += predicted.eq(target_indices).sum().item()
                else:
                    loss = self.criterion(outputs, targets)
                    _, predicted = outputs.max(1)
                    correct += predicted.eq(targets).sum().item()

                running_loss += loss.item() * imgs.size(0)
                total += imgs.size(0)

        val_loss = running_loss / total
        val_acc = correct / total
        return val_loss, val_acc

    def fit(self, epochs=10, early_stopping_patience=5):
        history = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
            "lr": []
        }

        best_val_loss = float("inf")
        patience_counter = 0
        best_model_path = os.path.join(self.checkpoint_dir, f"{self.model_name}_best.pth")

        start_time = time.time()
        print(f"\n=======================================================")
        print(f" Starting Training: {self.model_name} ({epochs} Epochs)")
        print(f"=======================================================")

        for epoch in range(1, epochs + 1):
            train_loss, train_acc = self.train_epoch()
            val_loss, val_acc = self.validate()

            current_lr = self.optimizer.param_groups[0]["lr"]
            if isinstance(self.scheduler, ReduceLROnPlateau):
                self.scheduler.step(val_loss)
            elif self.scheduler is not None:
                self.scheduler.step()

            history["train_loss"].append(train_loss)
            history["train_acc"].append(train_acc)
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)
            history["lr"].append(current_lr)

            print(f"Epoch [{epoch:02d}/{epochs:02d}] | Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}% | Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:.2f}% | LR: {current_lr:.6f}")

            # Checkpoint saving
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "val_loss": val_loss,
                    "val_acc": val_acc
                }, best_model_path)
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    print(f" Early stopping triggered after {epoch} epochs.")
                    break

        total_time = time.time() - start_time
        print(f" Training complete for {self.model_name} in {total_time/60:.2f} minutes.")
        print(f" Best Val Loss: {best_val_loss:.4f} | Weights saved to: {best_model_path}")
        return history, best_model_path
