"""Roomy, modeless sensor settings with one section visible at a time."""
from .qt import QtCore, QtWidgets


class SectionNavigation(QtWidgets.QListWidget):
    """Keep complete section names visible at different Windows text scales."""
    def __init__(self):
        super().__init__()
        self.setWordWrap(True)
        self.setTextElideMode(QtCore.Qt.ElideNone)
        self.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)

    def fit_labels(self):
        # Allow padding, item spacing and a vertical scrollbar at larger fonts.
        text_width = max((self.fontMetrics().horizontalAdvance(self.item(i).text())
                          for i in range(self.count())), default=0)
        self.setFixedWidth(max(320, text_width + 60))
        width = max(60, self.viewport().width()-30)
        for index in range(self.count()):
            item = self.item(index)
            bounds = self.fontMetrics().boundingRect(
                QtCore.QRect(0, 0, width, 1000), QtCore.Qt.TextWordWrap, item.text())
            item.setSizeHint(QtCore.QSize(0, bounds.height()+28))

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.fit_labels()

    def changeEvent(self, event):
        super().changeEvent(event)
        if event.type() in (QtCore.QEvent.FontChange, QtCore.QEvent.StyleChange):
            self.fit_labels()


class SettingsWindow(QtWidgets.QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Sensor settings")
        self.setWindowFlags(QtCore.Qt.Window | QtCore.Qt.WindowTitleHint |
                            QtCore.Qt.WindowSystemMenuHint | QtCore.Qt.WindowMinMaxButtonsHint |
                            QtCore.Qt.WindowCloseButtonHint)
        self.setMinimumSize(800, 580)
        screen = self.screen().availableGeometry()
        self.resize(min(1120, screen.width()-60), min(850, screen.height()-80))
        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(20)
        self.navigation = SectionNavigation()
        self.navigation.setSpacing(5)
        self.navigation.setStyleSheet("QListWidget { border: 0; } QListWidget::item { padding: 12px 10px; }")
        layout.addWidget(self.navigation)
        content = QtWidgets.QWidget()
        content_layout = QtWidgets.QVBoxLayout(content)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(12)
        self.heading = QtWidgets.QLabel()
        self.heading.setWordWrap(True)
        font = self.heading.font()
        font.setPointSize(font.pointSize()+4)
        font.setBold(True)
        self.heading.setFont(font)
        content_layout.addWidget(self.heading)
        hint = QtWidgets.QLabel("Changes apply immediately. Use Saved sensor configs to keep your settings.")
        self.hint = hint
        hint.setWordWrap(True)
        content_layout.addWidget(hint)
        self.pages = QtWidgets.QStackedWidget()
        content_layout.addWidget(self.pages, 1)
        layout.addWidget(content, 1)
        self.navigation.currentRowChanged.connect(self.select_page)

    def add_page(self, title, widget):
        if isinstance(widget, QtWidgets.QGroupBox):
            widget.setTitle("")
            widget.setFlat(True)
        if widget.layout() is not None:
            widget.layout().setContentsMargins(16, 16, 16, 16)
            widget.layout().setSpacing(12)
            if isinstance(widget.layout(), QtWidgets.QFormLayout):
                widget.layout().setFieldGrowthPolicy(QtWidgets.QFormLayout.AllNonFixedFieldsGrow)
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        container = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(widget)
        layout.addStretch()
        scroll.setWidget(container)
        self.pages.addWidget(scroll)
        self.navigation.addItem(title)
        self.navigation.fit_labels()
        if self.navigation.currentRow() < 0:
            self.navigation.setCurrentRow(0)

    def select_page(self, index):
        if index >= 0:
            self.pages.setCurrentIndex(index)
            self.heading.setText(self.navigation.item(index).text())
            self.hint.setText("Calibration uses a separate session. Save a fitted config, then select it to apply."
                              if self.navigation.item(index).text() == "Calibrate" else
                              "Changes apply immediately. Use Saved sensor configs to keep your settings.")
