from pathlib import Path

import numpy as np

from mesh2tact.prediction import ModelCandidate, architecture_name, discover_models, read_labels


def test_discover_models_recurses_and_uses_nearest_labels(tmp_path):
    root = tmp_path / 'models'
    nested = root / 'runs' / 'resnet50'
    nested.mkdir(parents=True)
    (nested / 'best.pth').write_bytes(b'checkpoint')
    (nested / 'labels.txt').write_text('sphere\ncube\n', encoding='utf-8')
    missing = root / 'other.pth'
    missing.write_bytes(b'checkpoint')

    found = discover_models(root)

    assert [candidate.checkpoint for candidate in found] == [missing, nested / 'best.pth']
    assert found[0].ready is False
    assert found[1].labels == ('sphere', 'cube')
    assert found[1].labels_path == nested / 'labels.txt'


def test_read_labels_rejects_duplicates(tmp_path):
    labels = tmp_path / 'labels.txt'
    labels.write_text('sphere\nsphere\n', encoding='utf-8')
    try:
        read_labels(labels)
    except ValueError as error:
        assert 'duplicate' in str(error)
    else:
        raise AssertionError('duplicate labels must be rejected')


def test_architecture_name_supports_the_training_families(tmp_path):
    labels = ('one', 'two')
    for name in ('resnet50', 'densenet121', 'efficientnet_b0', 'swin_t', 'vit_b_16'):
        candidate = ModelCandidate(tmp_path / f'best_{name}_model.pth', None, labels)
        state = {'fc.weight': object()} if name.startswith('resnet') else {}
        assert architecture_name(candidate, state) == name


def test_predict_classifier_loads_standard_checkpoint(tmp_path):
    torch = __import__('pytest').importorskip('torch')
    models = __import__('pytest').importorskip('torchvision.models', fromlist=['models'])
    from mesh2tact.prediction import ModelCandidate, predict_classifier

    checkpoint = tmp_path / 'resnet18.pth'
    model = models.resnet18(weights=None, num_classes=2)
    torch.save(model.state_dict(), checkpoint)
    labels_path = tmp_path / 'labels.txt'
    labels_path.write_text('flat\ncurved\n', encoding='utf-8')
    candidate = ModelCandidate(checkpoint, labels_path, ('flat', 'curved'))

    result = predict_classifier(np.zeros((80, 100, 3), dtype=np.uint8), candidate)

    assert result.label in candidate.labels
    assert len(result.top_k) == 2
    assert sum(confidence for _, confidence in result.top_k) == __import__('pytest').approx(1.0)
