import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
os.environ["QT_QPA_PLATFORM"] = "offscreen"
os.environ["XDG_RUNTIME_DIR"] = "/tmp/runtime-codespace"
from PyQt5.QtWidgets import QApplication
from machine_code_project.gui.gui_main import MainWindow
import sys

def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
