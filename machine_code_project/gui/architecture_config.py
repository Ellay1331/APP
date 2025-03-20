# -*- coding: utf-8 -*-
from PyQt5.QtWidgets import QWidget, QHBoxLayout, QLabel, QComboBox, QSpinBox

class ArchitectureConfigWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()
        
    def init_ui(self):
        layout = QHBoxLayout()
        self.config_combo = QComboBox()
        self.config_combo.addItems(["Small", "Medium", "Large", "Custom"])
        self.config_combo.setToolTip("Выберите предустановку или Custom для ручного ввода")
        layout.addWidget(QLabel("Конфигурация:"))
        layout.addWidget(self.config_combo)
        self.embed_spin = QSpinBox()
        self.embed_spin.setRange(64, 1024)
        self.embed_spin.setValue(256)
        self.embed_spin.setToolTip("Размер эмбеддинга")
        layout.addWidget(QLabel("Embed:"))
        layout.addWidget(self.embed_spin)
        self.nhead_spin = QSpinBox()
        self.nhead_spin.setRange(1, 16)
        self.nhead_spin.setValue(8)
        self.nhead_spin.setToolTip("Количество голов")
        layout.addWidget(QLabel("Heads:"))
        layout.addWidget(self.nhead_spin)
        self.layers_spin = QSpinBox()
        self.layers_spin.setRange(1, 12)
        self.layers_spin.setValue(6)
        self.layers_spin.setToolTip("Количество слоёв")
        layout.addWidget(QLabel("Layers:"))
        layout.addWidget(self.layers_spin)
        self.hidden_spin = QSpinBox()
        self.hidden_spin.setRange(64, 1024)
        self.hidden_spin.setValue(128)
        self.hidden_spin.setToolTip("Размер скрытого слоя")
        layout.addWidget(QLabel("Hidden:"))
        layout.addWidget(self.hidden_spin)
        self.config_combo.currentTextChanged.connect(self.update_config)
        self.update_config(self.config_combo.currentText())
        self.setLayout(layout)
        
    def update_config(self, text):
        if text == "Small":
            self.embed_spin.setValue(128)
            self.nhead_spin.setValue(4)
            self.layers_spin.setValue(2)
            self.hidden_spin.setValue(64)
            self.set_custom_enabled(False)
        elif text == "Medium":
            self.embed_spin.setValue(256)
            self.nhead_spin.setValue(8)
            self.layers_spin.setValue(6)
            self.hidden_spin.setValue(128)
            self.set_custom_enabled(False)
        elif text == "Large":
            self.embed_spin.setValue(512)
            self.nhead_spin.setValue(16)
            self.layers_spin.setValue(12)
            self.hidden_spin.setValue(256)
            self.set_custom_enabled(False)
        else:
            self.set_custom_enabled(True)
            
    def set_custom_enabled(self, enabled):
        self.embed_spin.setEnabled(enabled)
        self.nhead_spin.setEnabled(enabled)
        self.layers_spin.setEnabled(enabled)
        self.hidden_spin.setEnabled(enabled)
        
    def get_config(self):
        return {
            "embed_size": self.embed_spin.value(),
            "nhead": self.nhead_spin.value(),
            "num_layers": self.layers_spin.value(),
            "hidden_dim": self.hidden_spin.value()
        }
