import contextlib
import os
import pickle
import sys

import torch


def normalize_label(label: str) -> str:
    return str(label).strip().lower().replace(" ", "_")


def ensure_abs(path: str) -> str:
    return os.path.abspath(os.path.expanduser(path))


_SCENE_IO_NOISE = (
    "Reading camera",
    "[split] train list source",
    "Training images:",
    "Testing images:",
    "Loading Training Cameras",
    "Loading Test Cameras",
    "Number of points at initialisation",
)


@contextlib.contextmanager
def quiet_scene_io():
    stdout = sys.stdout

    class _Filter:
        def write(self, data):
            text = str(data)
            if (not text.strip()) or any(token in text for token in _SCENE_IO_NOISE):
                return len(data)
            return stdout.write(data)

        def flush(self):
            stdout.flush()

        def fileno(self):
            return stdout.fileno()

        def isatty(self):
            return stdout.isatty()

    sys.stdout = _Filter()
    try:
        yield
    finally:
        sys.stdout = stdout


def load_codebook(classifier_path: str, device: str = "cuda") -> torch.Tensor:
    try:
        ckpt = torch.load(classifier_path, map_location=device, weights_only=True)
    except (TypeError, pickle.UnpicklingError, RuntimeError):
        ckpt = torch.load(classifier_path, map_location=device, weights_only=False)
    if isinstance(ckpt, dict):
        codebook = ckpt["weight"].squeeze()
    else:
        codebook = ckpt
    return codebook.detach().clone()
