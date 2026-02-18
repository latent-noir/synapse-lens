"""Training view — controls for model/dataset selection and training parameters."""

from __future__ import annotations

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QLabel,
    QComboBox, QSpinBox, QDoubleSpinBox, QPushButton,
    QProgressBar, QTextEdit, QLineEdit, QFormLayout,
)
from PyQt5.QtCore import Qt

from nnviz.gui.models.app_state import AppState
from nnviz.engine.models import ModelFactory
from nnviz.data.registry import DatasetRegistry


class TrainingView(QWidget):
    """Training configuration and control panel."""

    def __init__(self, state: AppState, parent=None):
        super().__init__(parent)
        self.state = state
        self._build_ui()
        self._connect_signals()

    def _build_ui(self):
        layout = QVBoxLayout(self)

        # ── Model & Dataset Selection ────────────────────────────────────
        selector_group = QGroupBox("Model & Dataset")
        selector_form = QFormLayout()

        self.model_combo = QComboBox()
        self.model_combo.addItems(ModelFactory.available())
        selector_form.addRow("Architecture:", self.model_combo)

        self.dataset_combo = QComboBox()
        self.dataset_combo.addItems(DatasetRegistry.available())
        selector_form.addRow("Dataset:", self.dataset_combo)

        selector_group.setLayout(selector_form)
        layout.addWidget(selector_group)

        # ── Hyperparameters ──────────────────────────────────────────────
        hp_group = QGroupBox("Hyperparameters")
        hp_form = QFormLayout()

        self.lr_spin = QDoubleSpinBox()
        self.lr_spin.setRange(0.00001, 1.0)
        self.lr_spin.setDecimals(5)
        self.lr_spin.setSingleStep(0.0001)
        self.lr_spin.setValue(0.001)
        hp_form.addRow("Learning Rate:", self.lr_spin)

        self.epochs_spin = QSpinBox()
        self.epochs_spin.setRange(1, 500)
        self.epochs_spin.setValue(10)
        hp_form.addRow("Epochs:", self.epochs_spin)

        self.batch_spin = QSpinBox()
        self.batch_spin.setRange(8, 512)
        self.batch_spin.setSingleStep(8)
        self.batch_spin.setValue(64)
        hp_form.addRow("Batch Size:", self.batch_spin)

        self.dropout_spin = QDoubleSpinBox()
        self.dropout_spin.setRange(0.0, 0.9)
        self.dropout_spin.setDecimals(2)
        self.dropout_spin.setSingleStep(0.05)
        self.dropout_spin.setValue(0.2)
        hp_form.addRow("Dropout:", self.dropout_spin)

        self.hidden_edit = QLineEdit("256,128")
        self.hidden_edit.setPlaceholderText("e.g. 256,128,64")
        hp_form.addRow("Hidden Layers:", self.hidden_edit)

        hp_group.setLayout(hp_form)
        layout.addWidget(hp_group)

        # ── Controls ─────────────────────────────────────────────────────
        ctrl_layout = QHBoxLayout()

        self.train_btn = QPushButton("▶ Train")
        self.train_btn.setStyleSheet(
            "QPushButton { background: #27ae60; color: white; padding: 8px 20px; "
            "border-radius: 4px; font-weight: bold; }"
            "QPushButton:hover { background: #2ecc71; }"
        )
        ctrl_layout.addWidget(self.train_btn)

        self.stop_btn = QPushButton("⏹ Stop")
        self.stop_btn.setEnabled(False)
        self.stop_btn.setStyleSheet(
            "QPushButton { background: #e74c3c; color: white; padding: 8px 20px; "
            "border-radius: 4px; font-weight: bold; }"
            "QPushButton:hover { background: #c0392b; }"
            "QPushButton:disabled { background: #555; }"
        )
        ctrl_layout.addWidget(self.stop_btn)

        self.save_btn = QPushButton("💾 Save for Comparison")
        self.save_btn.setStyleSheet(
            "QPushButton { background: #3498db; color: white; padding: 8px 20px; "
            "border-radius: 4px; font-weight: bold; }"
            "QPushButton:hover { background: #2980b9; }"
        )
        ctrl_layout.addWidget(self.save_btn)

        layout.addLayout(ctrl_layout)

        # ── Progress ─────────────────────────────────────────────────────
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        self.progress_bar.setStyleSheet(
            "QProgressBar { background: #1a1a2e; border: 1px solid #333; "
            "border-radius: 4px; text-align: center; color: white; }"
            "QProgressBar::chunk { background: #0ff; border-radius: 3px; }"
        )
        layout.addWidget(self.progress_bar)

        # ── Training Log ─────────────────────────────────────────────────
        self.log_output = QTextEdit()
        self.log_output.setReadOnly(True)
        self.log_output.setMaximumHeight(200)
        self.log_output.setStyleSheet(
            "QTextEdit { background: #0d1117; color: #0ff; font-family: monospace; "
            "border: 1px solid #333; padding: 8px; }"
        )
        layout.addWidget(self.log_output)

        # ── Global Styling ───────────────────────────────────────────────
        self.setStyleSheet("""
            QGroupBox { color: #e0e0e0; font-weight: bold; border: 1px solid #333;
                       border-radius: 4px; padding-top: 16px; margin-top: 8px; }
            QGroupBox::title { subcontrol-origin: margin; padding: 0 4px; }
            QLabel { color: #b0b0b0; }
            QComboBox, QSpinBox, QDoubleSpinBox, QLineEdit {
                background: #16213e; color: #e0e0e0; border: 1px solid #333;
                padding: 4px; border-radius: 3px; }
        """)

    def _connect_signals(self):
        self.model_combo.currentTextChanged.connect(
            lambda v: setattr(self.state, 'model_name', v))
        self.dataset_combo.currentTextChanged.connect(
            lambda v: setattr(self.state, 'dataset_name', v))

    def log(self, message: str):
        """Append a message to the training log."""
        self.log_output.append(message)

    def get_config(self) -> dict:
        """Read current widget values into a config dict."""
        hidden = [int(x.strip()) for x in self.hidden_edit.text().split(",") if x.strip()]
        return {
            "learning_rate": self.lr_spin.value(),
            "epochs": self.epochs_spin.value(),
            "batch_size": self.batch_spin.value(),
            "dropout": self.dropout_spin.value(),
            "hidden_sizes": hidden,
        }
