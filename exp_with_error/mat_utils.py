from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Dict

import numpy as np


@dataclass
class MatData:
    raw: Dict[str, Any]


def _to_namespace(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        if obj.dtype == np.object_:
            return np.vectorize(_to_namespace, otypes=[object])(obj)
        return obj
    if hasattr(obj, "_fieldnames"):
        data = {name: _to_namespace(getattr(obj, name)) for name in obj._fieldnames}
        return SimpleNamespace(**data)
    return obj


def load_mat(path: str) -> MatData:
    try:
        import scipy.io as sio
    except ImportError as exc:
        raise ImportError("scipy is required to read .mat files") from exc

    try:
        raw = sio.loadmat(path, struct_as_record=False, squeeze_me=True)
    except NotImplementedError as exc:
        raise NotImplementedError(
            "Unsupported .mat version. Please re-save the file with MATLAB v7 or earlier."
        ) from exc

    data: Dict[str, Any] = {}
    for key, value in raw.items():
        if key.startswith("__"):
            continue
        data[key] = _to_namespace(value)
    return MatData(raw=data)


def as_array(value: Any) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return value
    return np.asarray(value)


def as_1d(value: Any) -> np.ndarray:
    arr = as_array(value).astype(float)
    return np.atleast_1d(arr).squeeze()


def as_2d(value: Any) -> np.ndarray:
    arr = as_array(value).astype(float)
    if arr.ndim == 1:
        return arr.reshape(-1, 1)
    return arr
