import os
import json
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
from mesh2tact.gui.qt import QtWidgets
from mesh2tact.gui.preview_grid import PreviewGrid


def test_preview_choices_restore_including_none(tmp_path):
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    path = tmp_path/'choices.json'
    first = PreviewGrid(QtWidgets.QLabel, path)
    assert first.selected() == ['rgb', 'depth']
    first.checkboxes['contact'].setChecked(True)
    first.checkboxes['depth'].setChecked(False)
    assert first.selected() == ['rgb', 'depth']
    assert not path.exists()
    first.apply_button.click()
    restored = PreviewGrid(QtWidgets.QLabel, path)
    assert restored.selected() == ['rgb', 'contact']
    restored.set_selected([])
    empty = PreviewGrid(QtWidgets.QLabel, path)
    assert empty.selected() == []
    assert json.loads(path.read_text())['panels'] == []
    for panel in (first, restored, empty):
        panel.close()


def test_invalid_preferences_preserved_until_user_changes_choice(tmp_path):
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    path = tmp_path/'choices.json'
    path.write_text('{invalid')
    panel = PreviewGrid(QtWidgets.QLabel, path)
    assert panel.selected() == ['rgb', 'depth']
    assert path.read_text() == '{invalid'
    panel.checkboxes['clean'].setChecked(True)
    assert path.read_text() == '{invalid'
    panel.apply_button.click()
    assert json.loads(path.read_text())['panels'] == ['rgb', 'depth', 'clean']
    panel.close()


def test_apply_failure_preserves_last_applied_selection(tmp_path):
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    panel = PreviewGrid(QtWidgets.QLabel, tmp_path/'saved.json')
    assert panel.set_selected(['rgb'])
    blocker = tmp_path/'not_a_directory'
    blocker.write_text('keep')
    panel.preferences_path = blocker/'choices.json'
    panel.checkboxes['contact'].setChecked(True)
    panel.apply_button.click()
    assert panel.selected() == ['rgb']
    assert panel.checkboxes['contact'].isChecked()
    assert 'could not be saved' in panel.persistence_status.text()
    assert json.loads((tmp_path/'saved.json').read_text())['panels'] == ['rgb']
    panel.close()


def test_tactile_image_is_first_after_loading_older_order(tmp_path):
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    path = tmp_path/'choices.json'
    path.write_text(json.dumps(dict(version=1, panels=['depth', 'rgb', 'contact'])))
    panel = PreviewGrid(QtWidgets.QLabel, path)
    assert next(iter(panel.checkboxes)) == 'rgb'
    assert panel.checkboxes['rgb'].text() == 'Tactile image'
    assert panel.selected() == ['rgb', 'depth', 'contact']
    assert panel.grid.itemAtPosition(0, 0).widget() is panel.cards['rgb']
    panel.close()
