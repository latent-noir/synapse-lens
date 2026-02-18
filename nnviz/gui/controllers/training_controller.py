"""Training controller — orchestrates model training in a background thread."""

from __future__ import annotations
from typing import Optional

from PyQt5.QtCore import QThread, pyqtSignal
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from nnviz.gui.models.app_state import AppState
from nnviz.engine.models import ModelFactory
from nnviz.engine.trainer import TrainerConfig


class TrainWorker(QThread):
    """Background thread for model training."""

    epoch_done = pyqtSignal(int, dict)   # epoch, metrics
    finished = pyqtSignal(dict)          # full history
    log_message = pyqtSignal(str)

    def __init__(self, model, config: dict, parent=None):
        super().__init__(parent)
        self.model = model
        self.config = config
        self._stop = False

    def run(self):
        epochs = self.config.get("epochs", 10)
        lr = self.config.get("learning_rate", 0.001)
        batch_size = self.config.get("batch_size", 64)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()

        # Demo data (in production, load from DatasetRegistry)
        if hasattr(self.model, 'features'):  # CNN
            X = torch.randn(256, 1, 28, 28)
        elif any('batch' in str(type(m)) for m in self.model.modules()):
            X = torch.randn(256, 41)  # AnomalyMLP
        else:
            X = torch.randn(256, 784)  # MLP
        num_classes = list(self.model.parameters())[-1].shape[0]
        y = torch.randint(0, num_classes, (256,))
        loader = DataLoader(TensorDataset(X, y), batch_size=batch_size, shuffle=True)

        history = {"train_loss": [], "train_accuracy": []}

        for epoch in range(epochs):
            if self._stop:
                self.log_message.emit(f"⏹ Stopped at epoch {epoch}")
                break

            self.model.train()
            epoch_loss, correct, total = 0.0, 0, 0

            for inputs, targets in loader:
                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                _, pred = outputs.max(1)
                correct += pred.eq(targets).sum().item()
                total += targets.size(0)

            avg_loss = epoch_loss / len(loader)
            accuracy = correct / total
            history["train_loss"].append(avg_loss)
            history["train_accuracy"].append(accuracy)

            metrics = {"loss": avg_loss, "accuracy": accuracy}
            self.epoch_done.emit(epoch, metrics)
            self.log_message.emit(
                f"Epoch {epoch+1}/{epochs} — Loss: {avg_loss:.4f} — Acc: {accuracy:.2%}"
            )

        self.finished.emit(history)

    def stop(self):
        self._stop = True


class TrainingController:
    """Controller connecting TrainingView to the training engine."""

    def __init__(self, state: AppState, view, parent=None):
        self.state = state
        self.view = view
        self.worker: Optional[TrainWorker] = None

        # Wire up buttons
        self.view.train_btn.clicked.connect(self._start_training)
        self.view.stop_btn.clicked.connect(self._stop_training)
        self.view.save_btn.clicked.connect(self._save_for_comparison)

    def _start_training(self):
        config = self.view.get_config()
        self.state.update_config(**config)

        # Create model
        model_kwargs = {"dropout": config["dropout"]}
        if self.state.model_name in ("mlp", "anomaly_mlp"):
            model_kwargs["hidden_sizes"] = config["hidden_sizes"]
        model = ModelFactory.create(self.state.model_name, **model_kwargs)

        self._model = model
        self.state.is_training = True
        self.view.train_btn.setEnabled(False)
        self.view.stop_btn.setEnabled(True)
        self.view.progress_bar.setValue(0)
        self.view.log("▶ Training started...")

        # Launch worker thread
        self.worker = TrainWorker(model, config)
        self.worker.epoch_done.connect(self._on_epoch_done)
        self.worker.finished.connect(self._on_finished)
        self.worker.log_message.connect(self.view.log)
        self.worker.start()

    def _stop_training(self):
        if self.worker:
            self.worker.stop()

    def _on_epoch_done(self, epoch: int, metrics: dict):
        total_epochs = self.state.config.get("epochs", 10)
        progress = int((epoch + 1) / total_epochs * 100)
        self.view.progress_bar.setValue(progress)
        self.state.status_updated.emit(
            f"Epoch {epoch+1}/{total_epochs} — Loss: {metrics['loss']:.4f}"
        )

    def _on_finished(self, history: dict):
        self.state.is_training = False
        self.state.history = history
        self.view.train_btn.setEnabled(True)
        self.view.stop_btn.setEnabled(False)
        self.view.progress_bar.setValue(100)
        self.state.status_updated.emit("✅ Training complete!")
        self.state.training_completed.emit(history)

    def _save_for_comparison(self):
        if hasattr(self, '_model') and self.state.history:
            name = f"{self.state.model_name} (lr={self.state.config.get('learning_rate', '?')})"
            self.state.add_comparison_model(name, self._model, self.state.history)
            self.state.status_updated.emit(f"Saved '{name}' for comparison")
