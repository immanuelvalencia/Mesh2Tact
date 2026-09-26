"""Real camera masking and raw image storage, independent of Qt/3D."""
from pathlib import Path
import re
import cv2
import numpy as np


DEFAULTS = dict(source=0, frame_scale=0.3, abs_thresh=0.01, abs_blur=10,
                close_size=11, open_size=5, auto_capture_threshold=3000,
                dataset_dir=str(Path(__file__).resolve().parents[1] / 'data' / 'gelsight'),
                label='object')


def contact_mask(frame, reference, threshold, blur, close_size=11, open_size=5):
    if frame.shape != reference.shape:
        raise ValueError('Frame size changed. Capture a new reference.')
    diff = cv2.absdiff(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY),
                       cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY))
    if blur > 1:
        blur = int(blur) | 1
        diff = cv2.GaussianBlur(diff, (blur, blur), 0)
    mask = cv2.threshold(diff, int(threshold * 255), 255, cv2.THRESH_BINARY)[1]
    for operation, size in ((cv2.MORPH_CLOSE, close_size), (cv2.MORPH_OPEN, open_size)):
        if size > 1:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (int(size) | 1,) * 2)
            mask = cv2.morphologyEx(mask, operation, kernel)
    return mask


def label_path(root, label):
    label = label.strip()
    if (not label or label in {'.', '..'} or re.search(r'[<>:"/\\|?*\x00-\x1f]', label)
            or label.endswith(('.', ' '))
            or label.split('.')[0].upper() in {'CON', 'PRN', 'AUX', 'NUL',
                *('COM%d' % i for i in range(1, 10)), *('LPT%d' % i for i in range(1, 10))}):
        raise ValueError('Enter a valid object label (one folder name).')
    if not str(root).strip():
        raise ValueError('Select a dataset folder.')
    return Path(root).expanduser().resolve() / label


class RawImageWriter:
    """Save qualifying camera frames directly in the class folder."""
    def __init__(self, directory):
        self.directory = Path(directory)
        self.index = None
        self.total = 0

    def accept(self, raw, mask, minimum):
        if np.count_nonzero(mask) < minimum:
            return None
        if self.index is None:
            self.directory.mkdir(parents=True, exist_ok=True)
            indices = [int(m.group(1)) for p in self.directory.iterdir()
                       if (m := re.fullmatch(r'(\d+)_raw\.png', p.name))]
            self.index = max(indices, default=-1) + 1
        ok, encoded = cv2.imencode('.png', raw)
        if not ok:
            raise OSError('PNG encoding failed')
        while True:
            path = self.directory / f'{self.index:06d}_raw.png'
            try:
                stream = path.open('xb')
                break
            except FileExistsError:
                self.index += 1
        try:
            with stream:
                stream.write(encoded.tobytes())
        except Exception:
            path.unlink(missing_ok=True)
            raise
        self.index += 1
        self.total += 1
        return path
