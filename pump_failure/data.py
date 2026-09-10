"""Chronological splits, training-only preprocessing, lazy continuous windows."""
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


def read_data(path: str, require_status: bool = True) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if "timestamp" not in frame:
        raise ValueError("CSV must contain a timestamp column.")
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="raise")
    if frame["timestamp"].isna().any():
        raise ValueError("Timestamps cannot be missing.")
    if not frame["timestamp"].is_monotonic_increasing or frame["timestamp"].duplicated().any():
        raise ValueError("Timestamps must be strictly increasing; fix unordered/duplicate rows.")
    sensors = [c for c in frame if c.startswith("sensor_")]
    if not sensors:
        raise ValueError("CSV must contain numeric sensor_* columns.")
    frame[sensors] = frame[sensors].apply(pd.to_numeric, errors="raise")
    frame[sensors] = frame[sensors].replace([np.inf, -np.inf], np.nan)
    if require_status and "machine_status" not in frame:
        raise ValueError("Training CSV requires machine_status labels.")
    if "machine_status" in frame:
        frame["machine_status"] = frame["machine_status"].astype("string").str.strip().str.upper()
        if require_status:
            unknown = ~frame["machine_status"].isin(["NORMAL", "BROKEN", "RECOVERING"])
            if unknown.any():
                raise ValueError("Training labels must be NORMAL, BROKEN, or RECOVERING.")
    return frame.reset_index(drop=True)


def split_data(frame: pd.DataFrame, train: float, validation: float,
               calibration: float) -> dict[str, pd.DataFrame]:
    fractions = np.array([train, validation, calibration], dtype=float)
    if not np.isfinite(fractions).all() or (fractions <= 0).any() or fractions.sum() >= 1:
        raise ValueError("Split fractions must be positive and sum to less than one.")
    cuts = [0, *[int(len(frame) * f) for f in np.cumsum(fractions)], len(frame)]
    names = ["train", "validation", "calibration", "test"]
    return {name: frame.iloc[a:b].reset_index(drop=True)
            for name, a, b in zip(names, cuts[:-1], cuts[1:])}


@dataclass
class Preprocessor:
    features: list[str]
    dropped_features: list[str]
    medians: list[float]
    means: list[float]
    scales: list[float]

    @classmethod
    def fit(cls, training: pd.DataFrame) -> "Preprocessor":
        normal = training.loc[training["machine_status"].eq("NORMAL")]
        if normal.empty:
            raise ValueError("Training split has no NORMAL rows.")
        sensors = [c for c in normal if c.startswith("sensor_")]
        dropped = [c for c in sensors if normal[c].isna().all()]
        features = [c for c in sensors if c not in dropped]
        if not features:
            raise ValueError("No observed sensors in normal training data.")
        observed = normal[features]
        medians = observed.median()
        filled = observed.fillna(medians)
        means = filled.mean()
        scales = filled.std(ddof=0)
        scales = scales.mask(scales < 1e-8, 1.0)
        return cls(features, dropped, medians.tolist(), means.tolist(), scales.tolist())

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        missing = set(self.features) - set(frame.columns)
        if missing:
            raise ValueError(f"Missing required sensor columns: {sorted(missing)}")
        values = frame[self.features].to_numpy(dtype=np.float64, copy=True)
        values = np.where(np.isfinite(values), values, np.asarray(self.medians))
        values = (values - np.asarray(self.means)) / np.asarray(self.scales)
        with np.errstate(over="ignore", invalid="ignore"):
            values = values.astype(np.float32)
        if not np.isfinite(values).all():
            raise ValueError("Preprocessed data is non-finite; inspect sensor magnitudes.")
        return values

    def to_dict(self) -> dict:
        return asdict(self)


class WindowDataset(Dataset):
    """Keep a single 2-D array; create windows only when requested.

    NORMAL filtering applies to complete windows, never to rows before windowing.
    This prevents concatenation across failures or timestamp gaps.
    """
    def __init__(self, values: np.ndarray, timestamps: pd.Series,
                 window_size: int = 150, stride: int = 1,
                 normal_mask: np.ndarray | None = None,
                 expected_interval_seconds: int = 60):
        if window_size < 1 or stride < 1 or expected_interval_seconds < 1:
            raise ValueError("Window size, stride, and interval must be positive.")
        if len(values) != len(timestamps):
            raise ValueError("Values and timestamps must have the same length.")
        self.values = torch.from_numpy(np.ascontiguousarray(values, dtype=np.float32))
        self.window_size = window_size
        starts = np.arange(0, max(0, len(values) - window_size + 1), stride, dtype=np.int64)
        ends = starts + window_size
        delta = timestamps.diff().dt.total_seconds().to_numpy()
        breaks = (delta != expected_interval_seconds).astype(np.int64)
        breaks[0:1] = 0
        cumulative = np.r_[0, np.cumsum(breaks)]
        # Gaps at the first row of a window are outside that window.
        valid = (cumulative[ends] - cumulative[starts + 1]) == 0
        if normal_mask is not None:
            if len(normal_mask) != len(values):
                raise ValueError("Normal mask must match the data length.")
            bad = np.r_[0, np.cumsum(~np.asarray(normal_mask, dtype=bool))]
            valid &= (bad[ends] - bad[starts]) == 0
        self.starts = starts[valid]

    def __len__(self) -> int:
        return len(self.starts)

    def __getitem__(self, index: int) -> torch.Tensor:
        start = int(self.starts[index])
        return self.values[start:start + self.window_size]
