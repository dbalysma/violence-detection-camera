"""Run live_camera.py headlessly for tests: fake model, fake camera, no window, no Ollama.

Used by tests/test_live_camera_headless.py in a subprocess. OpenCV's VideoWriter,
temporal_rule.py and live_options.py are the real ones; torch, the trained model, the camera
and Gemma are replaced. Run from a scratch directory: live_camera.py uses relative paths.

    python tests/_stub_live_camera.py <repo root> [live_camera.py arguments...]
"""

import contextlib
import runpy
import sys
import types

import cv2
import numpy as np

REPO = sys.argv[1]
sys.path.insert(0, REPO)


class FakeTensor:
    def __init__(self, array):
        self.array = np.asarray(array, dtype=float)

    def permute(self, *dims):
        return FakeTensor(self.array.transpose(dims))

    def unsqueeze(self, axis):
        return FakeTensor(np.expand_dims(self.array, axis))

    def to(self, _device):
        return self

    def __getitem__(self, index):
        return FakeTensor(self.array[index])

    def item(self):
        return float(self.array)


# One probability per clip: 2 low, then 4 violent (alert on the 3rd), then 2 low.
PROBABILITIES = [0.2, 0.3, 0.8, 0.8, 0.8, 0.8, 0.1, 0.1]
model_calls = []


def fake_model(_clip):
    p = PROBABILITIES[min(len(model_calls), len(PROBABILITIES) - 1)]
    model_calls.append(p)
    return FakeTensor([[np.log(1 - p), np.log(p)]])


def load_trained_model(_path, _device):
    print("STUB: model loaded", flush=True)
    return fake_model


def softmax(tensor, dim):
    e = np.exp(tensor.array - tensor.array.max(axis=dim, keepdims=True))
    return FakeTensor(e / e.sum(axis=dim, keepdims=True))


torch = types.ModuleType("torch")
torch.device = lambda name: name
torch.cuda = types.SimpleNamespace(is_available=lambda: False)
torch.stack = lambda tensors: FakeTensor(np.stack([t.array for t in tensors]))
torch.inference_mode = contextlib.nullcontext
torch.softmax = softmax

predict = types.ModuleType("predict")
predict.load_trained_model = load_trained_model
video_utils = types.ModuleType("video_utils")
video_utils.transform = lambda rgb: FakeTensor(cv2.resize(rgb, (8, 8)).transpose(2, 0, 1))
gemma_utils = types.ModuleType("gemma_utils")
gemma_utils.generate_report = lambda **_kw: "stub Gemma report"
sys.modules.update(torch=torch, predict=predict, video_utils=video_utils, gemma_utils=gemma_utils)

FRAMES = 16 + 8 * (len(PROBABILITIES) - 1)  # exactly one prediction per stride


class FakeCamera:
    def __init__(self, _index):
        print("STUB: camera opened", flush=True)
        self.frame = 0

    def isOpened(self):
        return True

    def get(self, _prop):
        return 30.0

    def read(self):
        if self.frame >= FRAMES:
            return False, None
        self.frame += 1
        return True, np.full((120, 160, 3), (self.frame * 3) % 255, np.uint8)

    def release(self):
        pass


cv2.VideoCapture = FakeCamera
cv2.imshow = lambda *_args: None
cv2.waitKey = lambda *_args: -1
cv2.destroyAllWindows = lambda: None

sys.argv = ["live_camera.py", *sys.argv[2:]]
runpy.run_path(f"{REPO}/live_camera.py", run_name="__main__")
print(f"STUB: {len(model_calls)} clips classified", flush=True)
