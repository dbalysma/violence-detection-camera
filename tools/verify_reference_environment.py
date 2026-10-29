"""Verify the prospective CPU reference environment without downloading model weights.

Run under Python 3.13 with requirements-reference-py313.txt installed.
No datasets, checkpoint, footage, Ollama or GPU are needed.
"""

import importlib.metadata
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PINS = ROOT / "requirements-reference-py313.txt"


def verify_versions():
    packages = {}
    for line in PINS.read_text(encoding="utf-8").splitlines():
        if not line or line.lstrip().startswith("#"):
            continue
        name, required = line.split("==", maxsplit=1)
        version = importlib.metadata.version(name).split("+", 1)[0]
        if version != required:
            raise RuntimeError(f"{name}: installed {version}; expected {required}")
        packages[name] = version
    return packages


def main():
    import sys
    if sys.version_info[:2] != (3, 13):
        raise RuntimeError("This reference pin set was tested on Python 3.13 only")
    versions = verify_versions()

    import cv2
    import matplotlib
    import numpy
    import pandas
    import sklearn
    import torch
    import torchvision
    from torchvision.models.video import r2plus1d_18

    model = r2plus1d_18(weights=None)
    params = sum(parameter.numel() for parameter in model.parameters())
    if params < 1_000_000:
        raise RuntimeError("R(2+1)D-18 model was not constructed as expected")
    print(f"PASS: Python {sys.version.split()[0]}, {len(versions)} package version pins")
    print(f"PASS: R(2+1)D-18 constructed without pretrained weight download ({params:,} parameters)")
    print(f"PASS: import smoke for cv2, numpy, pandas, sklearn, matplotlib, torch and torchvision")
    print(f"Build: torch={torch.__version__}, torchvision={torchvision.__version__}; cuda={torch.version.cuda}")


if __name__ == "__main__":
    main()
