# gui/tabs.py
# -*- coding: utf-8 -*-
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QTextEdit, QPushButton, QFileDialog, QComboBox, QSpinBox, QDoubleSpinBox, QProgressBar
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from machine_code_project.gui.architecture_config import ArchitectureConfigWidget
from machine_code_project.model import TransformerLM, generate_text
import torch

class CodeGenerationTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()
        
    def init_ui(self):
        layout = QVBoxLayout()
        model_layout = QHBoxLayout()
        self.model_file_edit = QLineEdit()
        self.model_file_edit.setToolTip("Выберите файл модели (.pt)")
        self.model_file_btn = QPushButton("Выбрать модель")
        self.model_file_btn.clicked.connect(self.browse_model_file)
        model_layout.addWidget(QLabel("Файл модели:"))
        model_layout.addWidget(self.model_file_edit)
        model_layout.addWidget(self.model_file_btn)
        layout.addLayout(model_layout)
        
        prompt_layout = QHBoxLayout()
        self.prompt_edit = QLineEdit()
        self.prompt_edit.setToolTip("Введите промт для генерации")
        prompt_layout.addWidget(QLabel("Промт:"))
        prompt_layout.addWidget(self.prompt_edit)
        layout.addLayout(prompt_layout)
        
        self.gen_btn = QPushButton("Сгенерировать код")
        self.gen_btn.clicked.connect(self.generate_code)
        layout.addWidget(self.gen_btn)
        
        self.gen_output = QTextEdit()
        self.gen_output.setReadOnly(True)
        layout.addWidget(self.gen_output)
        
        self.setLayout(layout)
        
    def browse_model_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "Выбрать модель", "", "Model Files (*.pt)")
        if path:
            self.model_file_edit.setText(path)
            
    def generate_code(self):
        model_path = self.model_file_edit.text()
        prompt = self.prompt_edit.text()
        if not model_path or not prompt:
            self.gen_output.setText("Пожалуйста, выберите модель и введите промт.")
            return
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = TransformerLM(VOCAB_SIZE, 256, 8, 512, 6).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device)['model_state_dict'])
        
        generated_code = generate_text(model, prompt, max_generate=100, device=device)
        self.gen_output.setText(generated_code)

class TrainingTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()
        
    def init_ui(self):
        layout = QVBoxLayout()
        # Здесь реализуйте GUI для обучения модели, например, поля ввода, лог, прогресс-бар и график.
        label = QLabel("TrainingTab - реализуйте обучение здесь.")
        layout.addWidget(label)
        self.setLayout(layout)

class FineTuningTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout()
        label = QLabel("FineTuningTab - реализуйте дообучение здесь.")
        layout.addWidget(label)
        self.setLayout(layout)

class FileConverterTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout()
        label = QLabel("FileConverterTab - реализуйте конвертацию файлов здесь.")
        layout.addWidget(label)
        self.setLayout(layout)

class WebParserTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout()
        label = QLabel("WebParserTab - реализуйте парсинг сайтов здесь.")
        layout.addWidget(label)
        self.setLayout(layout)

class RepoClonerTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout()
        label = QLabel("RepoClonerTab - реализуйте клонирование репозиториев здесь.")
        layout.addWidget(label)
        self.setLayout(layout)
