"""Keep GUI checks from overwriting the user's saved preview preferences."""
import pytest


@pytest.fixture(autouse=True)
def isolated_preview_preferences(tmp_path, monkeypatch):
    from mesh2tact.gui import preview_grid
    monkeypatch.setattr(preview_grid, 'PREFERENCES_PATH', tmp_path/'preview_preferences.json')
    from mesh2tact.gui import calibration
    monkeypatch.setattr(calibration, 'SESSION_DIR', tmp_path/'reference_collections')
