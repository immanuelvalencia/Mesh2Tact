def wait_capture(window):
    from mesh2tact.gui.qt import QtCore, QtWidgets
    if window.gather_worker is None:
        return
    loop = QtCore.QEventLoop()
    window.gather_worker.finished.connect(loop.quit)
    timer = QtCore.QTimer()
    timer.setSingleShot(True)
    timer.timeout.connect(loop.quit)
    timer.start(60000)
    loop.exec_()
    timer.stop()
    QtWidgets.QApplication.processEvents()
    assert window.gather_worker is None
