import os

if os.environ.get("CPU") and not os.environ["CPU"].lstrip("-").isdigit():
    os.environ.pop("CPU")

import numpy as np
from safetensors.numpy import load_file

from tinygrad import Tensor


def load_pth(path: str, device=None, dtype=None) -> dict[str, Tensor]:
    """Load a Real-ESRGAN .safetensors checkpoint into a dict of tinygrad tensors."""
    tensors = load_file(path)
    state: dict[str, Tensor] = {}
    for k, v in tensors.items():
        arr = np.asarray(v)
        if dtype is not None:
            arr = arr.astype(dtype)
        state[k] = Tensor(arr, device=device)
    return state