from __future__ import annotations

APP_STYLESHEET = """
QMainWindow { background: #10141a; color: #e6edf3; }
QWidget { font-family: "Segoe UI"; font-size: 13px; }
#topBar, #inspectorPanel, #inspectorScroll, #inspectorContents,
#videoPanel, #consolePanel, QListWidget, QTextEdit, QTableWidget {
    background: #151b22;
    border: 1px solid #2e3a46;
    border-radius: 6px;
    color: #e6edf3;
}
#inspectorScroll { border: none; }
#inspectorScroll QWidget { background: #151b22; color: #e6edf3; }
#videoPanel { background: #080c11; }
#panelTitle {
    border: none;
    color: #f6f8fa;
    font-weight: 700;
    padding: 4px 0 2px 0;
}
#statusGood { color: #46d369; font-weight: 700; }
#statusWarn { color: #f6c343; font-weight: 700; }
#statusBad { color: #ff6b6b; font-weight: 700; }
#statusMuted { color: #8b98a8; }
QLabel, QCheckBox { color: #e6edf3; }
QListWidget {
    outline: 0;
    padding: 4px;
}
QListWidget::item {
    border-radius: 4px;
    min-height: 24px;
    padding: 3px 6px;
}
QListWidget::item:selected { background: #1f6feb; color: #ffffff; }
QComboBox, QLineEdit, QSpinBox, QDoubleSpinBox {
    background: #0d141c;
    border: 1px solid #3d4c5f;
    border-radius: 5px;
    color: #f6f8fa;
    min-height: 28px;
    padding: 3px 8px;
}
QComboBox:disabled, QLineEdit:disabled, QSpinBox:disabled, QDoubleSpinBox:disabled {
    color: #7d8794;
    background: #111820;
}
QPushButton {
    background: #223044;
    border: 1px solid #40536b;
    border-radius: 6px;
    min-height: 28px;
    padding: 5px 10px;
    color: #f6f8fa;
}
QPushButton:hover { background: #2b3d55; }
QPushButton:disabled {
    background: #18212b;
    color: #77818e;
    border-color: #26313d;
}
#emergencyButton {
    background: #8b2525;
    border-color: #cf4a4a;
    font-weight: 700;
}
#emergencyButton:hover { background: #a62b2b; }
QHeaderView::section {
    background: #202a35;
    color: #e6edf3;
    border: 0;
    padding: 5px;
}
QTableWidget {
    gridline-color: #2e3a46;
    selection-background-color: #1f6feb;
}
QScrollBar:vertical {
    background: #111820;
    width: 12px;
    margin: 0;
}
QScrollBar::handle:vertical {
    background: #344458;
    min-height: 24px;
    border-radius: 5px;
}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
    height: 0;
}
"""
