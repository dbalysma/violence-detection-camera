"""Offline smoke tests for the PyTorch pipeline, using random weights and a synthetic video.

These run in the "Reference CPU environment verification" workflow, where torch,
torchvision and OpenCV are installed. Elsewhere they are skipped. They check that
the code runs end to end; they do not reproduce any dissertation result, use any
dataset, or need the private checkpoint.

Every pretrained-weight download is blocked in setUpModule, so an accidental
download makes the tests fail instead of reaching the network.
"""

import importlib.util
import tempfile
import unittest
from pathlib import Path

HAVE_STACK = all(
    importlib.util.find_spec(name) is not None for name in ("torch", "torchvision", "cv2")
)

if HAVE_STACK:
    import cv2
    import numpy as np
    import torch
    import torch.hub
    from torchvision.models import _api as torchvision_weights_api

    from model import create_model
    from predict import load_trained_model, predict_video
    from temporal_rule import ALERT_CONSECUTIVE_CLIPS
    from video_utils import create_clips, load_video

FRAMES = 40  # 40 frames -> clips start at 0, 8, 16, 24 -> 4 clips of 16 frames
_patched = []


class DownloadBlocked(RuntimeError):
    pass


def _refuse_download(*args, **kwargs):
    raise DownloadBlocked("A pretrained-weight download was attempted during offline tests")


def setUpModule():
    if not HAVE_STACK:
        return
    for owner, name in (
        (torchvision_weights_api.WeightsEnum, "get_state_dict"),
        (torch.hub, "load_state_dict_from_url"),
        (torch.hub, "download_url_to_file"),
    ):
        _patched.append((owner, name, getattr(owner, name)))
        setattr(owner, name, _refuse_download)


def tearDownModule():
    while _patched:
        owner, name, original = _patched.pop()
        setattr(owner, name, original)


def write_synthetic_video(path, frames=FRAMES, size=(160, 120)):
    """Write a short random-noise AVI. Returns False if this OpenCV build can't encode it."""
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 30, size)
    if not writer.isOpened():
        return False
    rng = np.random.default_rng(0)
    for _ in range(frames):
        writer.write(rng.integers(0, 256, (size[1], size[0], 3), dtype=np.uint8))
    writer.release()
    return path.is_file() and path.stat().st_size > 0


@unittest.skipUnless(HAVE_STACK, "torch, torchvision and OpenCV are not installed")
class OfflineModelTests(unittest.TestCase):
    def test_default_model_still_requests_pretrained_weights(self):
        # Proves both that the guard works and that training keeps Kinetics-400 initialisation.
        with self.assertRaises(DownloadBlocked):
            create_model()

    def test_untrained_model_outputs_two_classes_with_frozen_backbone(self):
        torch.manual_seed(0)
        model = create_model(pretrained=False).eval()
        with torch.inference_mode():
            output = model(torch.randn(2, 3, 16, 112, 112))
        self.assertEqual(tuple(output.shape), (2, 2))
        trainable = {name for name, p in model.named_parameters() if p.requires_grad}
        self.assertTrue(trainable)
        self.assertTrue(all(name.startswith("fc.") for name in trainable), trainable)

    def test_checkpoint_round_trip_reproduces_outputs(self):
        torch.manual_seed(1)
        model = create_model(pretrained=False).eval()
        clip = torch.randn(1, 3, 16, 112, 112)
        with tempfile.TemporaryDirectory() as folder:
            checkpoint = Path(folder) / "random.pth"
            torch.save(model.state_dict(), checkpoint)
            loaded = load_trained_model(checkpoint, torch.device("cpu"))
        with torch.inference_mode():
            self.assertTrue(torch.equal(model(clip), loaded(clip)))


@unittest.skipUnless(HAVE_STACK, "torch, torchvision and OpenCV are not installed")
class SyntheticVideoPipelineTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.video = Path(self.folder.name) / "synthetic.avi"
        if not write_synthetic_video(self.video):
            self.skipTest("This OpenCV build cannot write MJPG/AVI test videos")

    def tearDown(self):
        self.folder.cleanup()

    def test_video_is_split_into_16_frame_clips(self):
        frames = load_video(str(self.video))
        self.assertEqual(len(frames), FRAMES)
        self.assertEqual(tuple(frames[0].shape), (3, 112, 112))
        clips = create_clips(frames)
        self.assertEqual(len(clips), 4)
        self.assertEqual(tuple(clips[0].shape), (3, 16, 112, 112))

    def test_predict_video_end_to_end_with_random_checkpoint(self):
        torch.manual_seed(2)
        checkpoint = Path(self.folder.name) / "random.pth"
        torch.save(create_model(pretrained=False).state_dict(), checkpoint)

        result = predict_video(str(self.video), checkpoint_path=checkpoint, use_gemma=False)

        self.assertEqual(result["total_clips"], 4)
        self.assertAlmostEqual(result["video_duration"], FRAMES / 30)
        self.assertGreaterEqual(result["max_consecutive_violent_clips"], 0)
        self.assertLessEqual(result["max_consecutive_violent_clips"], 4)
        expected = (
            "Violence"
            if result["max_consecutive_violent_clips"] >= ALERT_CONSECUTIVE_CLIPS
            else "Non-Violence"
        )
        self.assertEqual(result["prediction"], expected)
        self.assertAlmostEqual(
            result["violence_probability"] + result["non_violence_probability"], 100.0
        )
        self.assertEqual(result["gemma_report"], "")

    def test_missing_checkpoint_fails_clearly(self):
        missing = Path(self.folder.name) / "does-not-exist.pth"
        with self.assertRaises(FileNotFoundError):
            predict_video(str(self.video), checkpoint_path=missing, use_gemma=False)


if __name__ == "__main__":
    unittest.main()
