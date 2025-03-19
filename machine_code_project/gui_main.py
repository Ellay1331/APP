# -*- coding: utf-8 -*-
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QTextEdit, QPushButton, QFileDialog, QComboBox, QSpinBox, QDoubleSpinBox, QProgressBar
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from tabs import CodeGenerationTab, TrainingTab, FineTuningTab, FileConverterTab, WebParserTab, RepoClonerTab

class CodeGenerationTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()
        
    def init_ui(self):
        layout = QVBoxLayout()
        
        # Модель
        model_layout = QHBoxLayout()
        self.model_file_edit = QLineEdit()
        self.model_file_edit.setToolTip("Выберите файл модели (.pt)")
        self.model_file_btn = QPushButton("Выбрать модель")
        self.model_file_btn.clicked.connect(self.browse_model_file)
        model_layout.addWidget(QLabel("Файл модели:"))
        model_layout.addWidget(self.model_file_edit)
        model_layout.addWidget(self.model_file_btn)
        layout.addLayout(model_layout)
        
        # Промт
        prompt_layout = QHBoxLayout()
        self.prompt_edit = QLineEdit()
        self.prompt_edit.setToolTip("Введите промт для генерации")
        prompt_layout.addWidget(QLabel("Промт:"))
        prompt_layout.addWidget(self.prompt_edit)
        layout.addLayout(prompt_layout)
        
        # Кнопка генерации и вывод
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
        # Здесь должна быть логика генерации кода
        self.gen_output.setText("Пример сгенерированного кода.")

class TrainingTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        # Минимальная заглушка; реализация аналогична CodeGenerationTab
        layout = QVBoxLayout()
        layout.addWidget(QLabel("TrainingTab - реализация"))
        self.setLayout(layout)

class FineTuningTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout()
        layout.addWidget(QLabel("FineTuningTab - реализация"))
        self.setLayout(layout)

class FileConverterTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout()
        layout.addWidget(QLabel("FileConverterTab - реализация"))
        self.setLayout(layout)

class WebParserTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout()
        layout.addWidget(QLabel("WebParserTab - реализация"))
        self.setLayout(layout)

class RepoClonerTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout()
        layout.addWidget(QLabel("RepoClonerTab - реализация"))
        self.setLayout(layout)
