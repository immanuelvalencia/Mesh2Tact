"""Qt binding discovery.

``qtpy`` is only a shim: it picks whichever real binding (PyQt5, PySide6, ...)
is installed and raises an unhelpful ``QtBindingsNotFoundError`` when there is
none.  This module finds a binding first, tells qtpy which one to use, and
otherwise fails with a message that says exactly what to install.
"""

from __future__ import annotations

import importlib
import os

BINDINGS = [("PyQt5", "pyqt5"), ("PySide6", "pyside6"),
            ("PyQt6", "pyqt6"), ("PySide2", "pyside2")]

INSTALL_HINT = (
    "No Qt binding is installed in this environment.\n"
    "\n"
    "    conda install -c conda-forge pyqt pyvistaqt\n"
    "\n"
    "or, if you would rather stay with pip:\n"
    "\n"
    "    pip install PyQt5 pyvistaqt\n"
    "\n"
    "qtpy on its own is not enough - it is a shim that needs one of PyQt5, "
    "PySide6, PyQt6 or PySide2 underneath it."
)


class MissingQtBinding(ImportError):
    """Raised when no usable Qt binding is present."""


def available_binding() -> str | None:
    """Name of the first importable Qt binding, or ``None``."""
    for module, api in BINDINGS:
        try:
            importlib.import_module(module)
        except ImportError:
            continue
        os.environ.setdefault("QT_API", api)
        return module
    return None


def load_qt():
    """Return ``(QtCore, QtGui, QtWidgets)`` or raise :class:`MissingQtBinding`."""
    if available_binding() is None:
        raise MissingQtBinding(INSTALL_HINT)
    try:
        from qtpy import QtCore, QtGui, QtWidgets
    except ImportError as exc:                  # qtpy itself absent
        raise MissingQtBinding(
            f"{exc}\n\n    conda install -c conda-forge qtpy\n"
        ) from exc
    return QtCore, QtGui, QtWidgets


def __getattr__(name: str):
    """Resolve the Qt modules on first use.

    Deliberately lazy: ``main.py --check`` imports :func:`available_binding` from
    this module to report what is missing, and that must not blow up on the very
    machines where Qt is absent.
    """
    if name in {"QtCore", "QtGui", "QtWidgets"}:
        modules = dict(zip(("QtCore", "QtGui", "QtWidgets"), load_qt()))
        globals().update(modules)
        return modules[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
