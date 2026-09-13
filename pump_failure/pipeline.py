"""Train, calibrate, and score the pump LSTM autoencoder."""
import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import random
import sys
from time import perf_counter

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .data import (Preprocessor, WindowDataset, binary_status_labels, calibration_normal_mask, read_data,
                   split_data)
from .model import LSTMAutoencoder, reconstruction_error, reconstruction_loss


@dataclass
class Config:
    data_path: str = "dataset/sensor.csv"
    output_dir: str = "artifacts/pump_lstm"
    window_size: int = 150
    stride: int = 1
    expected_interval_seconds: int = 60
    train_fraction: float = 0.60
    validation_fraction: float = 0.20
    calibration_fraction: float = 0.0  # Legacy checkpoint compatibility only.
    hidden_size: int = 64
    latent_size: int = 32
    dropout: float = 0.1
    batch_size: int = 64
    epochs: int = 50
    learning_rate: float = 0.001
    huber_delta: float = 1.0
    weight_decay: float = 0.00001
    patience: int = 7
    gradient_clip: float = 1.0
    threshold_quantile: float = 0.99
    calibration_cooldown_hours: float = 0.0
    seed: int = 42
    num_workers: int = 0
    cpu_threads: int = 4
    device: str = "auto"

    def validate(self):
        for key in ("window_size", "stride", "expected_interval_seconds", "hidden_size",
                    "latent_size", "batch_size", "epochs", "patience", "cpu_threads"):
            value = getattr(self, key)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{key} must be a positive integer.")
        if not isinstance(self.num_workers, int) or self.num_workers < 0:
            raise ValueError("num_workers must be a nonnegative integer.")
        if not 0 <= self.dropout < 1 or not 0 < self.threshold_quantile < 1:
            raise ValueError("dropout must be in [0,1); threshold_quantile in (0,1).")
        for name in ("learning_rate", "gradient_clip", "weight_decay", "huber_delta"):
            value = getattr(self, name)
            if not np.isfinite(value) or value < 0 or (name != "weight_decay" and value == 0):
                raise ValueError(f"Invalid {name}.")
        if not np.isfinite(self.calibration_cooldown_hours) or self.calibration_cooldown_hours < 0:
            raise ValueError("Calibration cooldown hours must be finite and nonnegative.")
        if self.device not in ("auto", "cpu", "cuda"):
            raise ValueError("device must be auto, cpu, or cuda.")


def select_device(name: str) -> torch.device:
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    if name == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable in this environment.")
    return torch.device(name)


def save_json(path: Path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def prepare(config: Config):
    config.validate()
    frame = read_data(config.data_path)
    splits = split_data(frame, config.train_fraction, config.validation_fraction,
                        config.calibration_fraction)
    processor = Preprocessor.fit(splits["train"])
    datasets = {}
    report = {"rows": len(frame), "features": processor.features,
              "dropped_features": processor.dropped_features, "splits": {}}
    calibration_mask = calibration_normal_mask(frame, config.calibration_cooldown_hours)
    offset = 0
    for name, part in splits.items():
        normal = part["machine_status"].eq("NORMAL").to_numpy(dtype=bool)
        if name == "calibration":
            normal &= calibration_mask[offset:offset + len(part)]
        offset += len(part)
        dataset = WindowDataset(processor.transform(part), part["timestamp"],
                                config.window_size, config.stride,
                                normal if name != "test" else None,
                                config.expected_interval_seconds)
        if not len(dataset):
            raise ValueError(f"No eligible {config.window_size}-row windows in {name} split.")
        datasets[name] = dataset
        report["splits"][name] = {
            "rows": len(part), "windows": len(dataset),
            "start": str(part["timestamp"].iloc[0]), "end": str(part["timestamp"].iloc[-1]),
            "statuses": {k: int(part.machine_status.eq(k).sum())
                         for k in ("NORMAL", "BROKEN", "RECOVERING")},
            "binary_labels": {str(k): int(part.binary_label.eq(k).sum()) for k in (0, 1)},
            "missing_fraction": {k: float(v) for k, v in part[processor.features].isna().mean().items()},
        }
    if "calibration" not in splits:
        part = splits["validation"]
        offset = len(splits["train"])
        datasets["calibration"] = WindowDataset(
            processor.transform(part), part.timestamp, config.window_size, config.stride,
            calibration_mask[offset:offset + len(part)], config.expected_interval_seconds)
        if not len(datasets["calibration"]):
            raise ValueError("No eligible validation windows for threshold calibration.")
    report["threshold_source"] = "calibration" if "calibration" in splits else "validation"
    report["threshold_windows"] = len(datasets["calibration"])
    return splits, datasets, processor, report


def make_loader(dataset, config, shuffle=False):
    return DataLoader(dataset, batch_size=config.batch_size, shuffle=shuffle,
                      num_workers=config.num_workers,
                      pin_memory=select_device(config.device).type == "cuda")


def progress_bar(iterable=None, *, total=None, description="", enabled=True,
                 position=0, leave=True):
    return tqdm(iterable, total=total, desc=description, unit="batch",
                disable=not enabled or not sys.stderr.isatty(),
                dynamic_ncols=True, ascii=True, mininterval=0.25, smoothing=0,
                position=position, leave=leave,
                bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} "
                           "[elapsed {elapsed} | ETA {remaining}{postfix}]")


def run_epoch(model, loader, device, optimizer=None, gradient_clip=1.0,
              description="", overall=None, show_progress=False, huber_delta=1.0):
    model.train(optimizer is not None)
    total, count = 0.0, 0
    with torch.set_grad_enabled(optimizer is not None), progress_bar(
            loader, description=description, enabled=show_progress,
            position=1 if overall is not None else 0, leave=False) as batches:
        for x in batches:
            x = x.to(device, non_blocking=True)
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
            loss = reconstruction_loss(x, model(x), delta=huber_delta)
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite reconstruction loss.")
            if optimizer is not None:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip,
                                               error_if_nonfinite=True)
                optimizer.step()
            total += loss.item() * len(x)
            count += len(x)
            batches.set_postfix(huber=f"{total / count:.6f}", refresh=False)
            if overall is not None:
                overall.update(1)
    return total / count


@torch.inference_mode()
def score(model, loader, device, description="Scoring", show_progress=False):
    model.eval()
    errors = []
    with progress_bar(loader, description=description, enabled=show_progress) as batches:
        for x in batches:
            x = x.to(device, non_blocking=True)
            errors.append(reconstruction_error(x, model(x)).cpu().numpy())
    if not errors:
        raise ValueError("No continuous windows available for scoring.")
    result = np.concatenate(errors)
    if not np.isfinite(result).all():
        raise RuntimeError("Non-finite reconstruction errors.")
    return result


def score_table(frame, dataset, errors, threshold=None):
    starts = dataset.starts
    ends = starts + dataset.window_size - 1
    result = pd.DataFrame({
        "window_start_row": starts,
        "window_end_row": ends,
        "window_start": frame["timestamp"].iloc[starts].to_numpy(),
        "window_end": frame["timestamp"].iloc[ends].to_numpy(),
        "reconstruction_mse": errors,
    })
    if threshold is not None:
        result["threshold"] = threshold
        result["is_anomaly"] = errors > threshold
    if "machine_status" in frame:
        result["end_status"] = frame["machine_status"].iloc[ends].to_numpy()
        result["binary_label"] = binary_status_labels(frame["machine_status"])[ends]
        abnormal = ~frame["machine_status"].eq("NORMAL").fillna(False).to_numpy(dtype=bool)
        cumulative = np.r_[0, np.cumsum(abnormal)]
        result["contains_abnormal_status"] = (cumulative[ends + 1] - cumulative[starts]) > 0
    return result


def map_reconstruction_errors(frame, dataset, errors, threshold):
    """Assign each valid window MSE to its final row; leave unscored rows zero."""
    errors = np.asarray(errors, dtype=float)
    if len(frame) != len(dataset.values):
        raise ValueError("Frame and window dataset must have the same number of rows.")
    if errors.shape != (len(dataset),) or not np.isfinite(errors).all() or (errors < 0).any():
        raise ValueError("Expected one finite, nonnegative error per valid window.")
    if not np.isfinite(threshold) or threshold < 0:
        raise ValueError("Threshold must be finite and nonnegative.")
    ends = dataset.starts + dataset.window_size - 1
    ignored_ends = dataset.ignored_starts + dataset.window_size - 1
    mapped = np.zeros(len(frame), dtype=float)
    mapped[ends] = errors
    status = np.full(len(frame), "not_selected", dtype=object)
    status[:dataset.window_size - 1] = "warmup"
    status[ignored_ends] = "ignored_window"
    status[ends] = "scored"
    result = pd.DataFrame({
        "row_index": np.arange(len(frame)),
        "timestamp": frame["timestamp"].to_numpy(),
        "reconstruction_mse": mapped,
        "score_status": status,
        "is_scored": status == "scored",
        "threshold": threshold,
        "is_anomaly": (status == "scored") & (mapped > threshold),
    })
    if "machine_status" in frame:
        result["machine_status"] = frame["machine_status"].to_numpy()
        result["binary_label"] = binary_status_labels(frame["machine_status"])
    return result


def save_point_scores(frame, dataset, errors, threshold, output_path):
    """Export row-aligned scores, threshold exceedances, and a time plot."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    points = map_reconstruction_errors(frame, dataset, errors, threshold)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    points.to_csv(output, index=False)
    anomalies = points.loc[points["is_anomaly"]]
    anomalies.to_csv(output.with_name(output.stem + "_exceedances.csv"), index=False)
    labeled = "machine_status" in points
    fig = Figure(figsize=(14, 7 if labeled else 5), layout="constrained")
    FigureCanvasAgg(fig)
    if labeled:
        ax, status_ax = fig.subplots(2, 1, sharex=True, gridspec_kw={"height_ratios": [4, 1]})
        for status, label, color in (("NORMAL", 0, "green"), ("BROKEN", 1, "purple"),
                                      ("RECOVERING", 1, "orange")):
            selected = points.machine_status.eq(status)
            status_ax.scatter(points.loc[selected, "timestamp"],
                              points.loc[selected, "binary_label"], s=5, color=color,
                              label=f"{status.title()} = {label}")
        status_ax.set(yticks=[0, 1], ylim=(-0.15, 1.15), ylabel="Actual label", xlabel="Time")
        status_ax.legend(loc="upper left", ncol=3, fontsize=8)
        status_ax.grid(alpha=0.2)
    else:
        ax = fig.subplots()
    ax.plot(points["timestamp"], points["reconstruction_mse"], linewidth=0.7,
            label="Window reconstruction MSE at final row")
    ax.axhline(threshold, color="darkorange", linestyle="--",
               label=f"Anomaly threshold = {threshold:.6g}")
    ax.scatter(anomalies["timestamp"], anomalies["reconstruction_mse"],
               color="red", s=9, zorder=3,
               label=f"Above threshold ({len(anomalies):,} points)")
    ax.set(xlabel="Time" if not labeled else "", ylabel="Mean squared reconstruction error",
           title="Pump inference: reconstruction error and threshold")
    ax.legend(loc="best")
    ax.grid(alpha=0.2)
    fig.autofmt_xdate()
    fig.savefig(output.with_suffix(".png"), dpi=160)
    return points


def train(config: Config, dry_run=False, show_progress=True):
    run_started = perf_counter()
    print("Preparing sensor data and continuous windows...", flush=True)
    config.validate()
    if (config.train_fraction, config.validation_fraction, config.calibration_fraction) != (0.6, 0.2, 0.0):
        raise ValueError("New training requires chronological train/validation/test fractions 0.6/0.2/0.2.")
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.set_num_threads(config.cpu_threads)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    device = select_device(config.device)
    splits, datasets, processor, report = prepare(config)
    architecture = {"input_size": len(processor.features), "hidden_size": config.hidden_size,
                    "latent_size": config.latent_size, "dropout": config.dropout}
    model = LSTMAutoencoder(**architecture).to(device)
    loaders = {name: make_loader(ds, config, name == "train") for name, ds in datasets.items()}
    print(json.dumps({"device": str(device), "sensors": len(processor.features),
                      "dropped": processor.dropped_features,
                      "windows": {name: len(ds) for name, ds in datasets.items()}}, indent=2),
          flush=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate,
                                  weight_decay=config.weight_decay)
    output = Path(config.output_dir)
    if dry_run:
        x = next(iter(loaders["train"])).to(device)
        reconstructed = model(x)
        loss = reconstruction_loss(x, reconstructed, delta=config.huber_delta)
        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite dry-run loss.")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip,
                                       error_if_nonfinite=True)
        optimizer.step()
        output.mkdir(parents=True, exist_ok=True)
        report["dry_run"] = {"input_shape": list(x.shape), "output_shape": list(reconstructed.shape),
                             "huber_loss": float(loss.item()), "device": str(device)}
        save_json(output / "dry_run_report.json", report)
        print("Dry run passed: forward, backward, optimizer step. No checkpoint saved.", flush=True)
        return report

    output.mkdir(parents=True, exist_ok=True)
    if (output / "best_model.pt").exists():
        raise FileExistsError("Output already contains best_model.pt; choose a new --output directory.")
    save_json(output / "config.json", asdict(config))
    save_json(output / "data_report.json", report)
    for name, part in splits.items():
        labels = part[["timestamp", "machine_status", "binary_label"]].copy()
        labels.insert(0, "row_index", np.arange(len(part)))
        labels.to_csv(output / f"{name}_labels.csv", index=False)
    save_json(output / "preprocessing.json", processor.to_dict())
    history, best, stale = [], float("inf"), 0
    training_started = perf_counter()
    print(f"Training for up to {config.epochs} epochs. ETA excludes final scoring; "
          "early stopping may finish sooner.", flush=True)
    with progress_bar(total=config.epochs * (len(loaders["train"]) + len(loaders["validation"])),
                      description="Overall training", enabled=show_progress) as overall:
        for epoch in range(1, config.epochs + 1):
            epoch_started = perf_counter()
            train_loss = run_epoch(
                model, loaders["train"], device, optimizer, config.gradient_clip,
                description=f"Epoch {epoch}/{config.epochs} train", overall=overall,
                show_progress=show_progress, huber_delta=config.huber_delta)
            val_loss = run_epoch(
                model, loaders["validation"], device,
                description=f"Epoch {epoch}/{config.epochs} validate", overall=overall,
                show_progress=show_progress, huber_delta=config.huber_delta)
            epoch_seconds = perf_counter() - epoch_started
            history.append({"epoch": epoch, "train_huber": train_loss, "validation_huber": val_loss,
                            "epoch_seconds": epoch_seconds,
                            "elapsed_seconds": perf_counter() - training_started})
            save_json(output / "history.json", history)
            elapsed = perf_counter() - training_started
            remaining = elapsed / epoch * (config.epochs - epoch)
            tqdm.write(f"Epoch {epoch}/{config.epochs}: train_huber={train_loss:.6f} "
                       f"validation_huber={val_loss:.6f} | epoch {tqdm.format_interval(epoch_seconds)} "
                       f"| elapsed {tqdm.format_interval(elapsed)} "
                       f"| ETA to max epochs {tqdm.format_interval(remaining)}")
            if val_loss < best:
                best, stale = val_loss, 0
                torch.save({"format_version": 1, "model_state": model.state_dict(),
                            "architecture": architecture, "config": asdict(config),
                            "preprocessing": processor.to_dict(), "epoch": epoch,
                            "validation_huber": best,
                            "training_objective": {"loss": "huber", "delta": config.huber_delta,
                                                   "optimizer": "Adam"},
                            "score_metric": "reconstruction_mse"}, output / "best_model.pt")
            else:
                stale += 1
                if stale >= config.patience:
                    overall.set_description("Training stopped early")
                    tqdm.write(f"Early stopping after {epoch} of {config.epochs} maximum epochs.")
                    break
    print(f"Training finished in {tqdm.format_interval(perf_counter() - training_started)}.",
          flush=True)
    checkpoint = torch.load(output / "best_model.pt", map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state"])
    errors = score(model, loaders["calibration"], device,
                   description="Calibration", show_progress=show_progress)
    threshold = float(np.quantile(errors, config.threshold_quantile))
    checkpoint["threshold"] = threshold
    torch.save(checkpoint, output / "best_model.pt")
    save_json(output / "threshold.json",
              {"mse_threshold": threshold, "quantile": config.threshold_quantile,
               "normal_calibration_windows": len(errors),
               "calibration_cooldown_hours": config.calibration_cooldown_hours,
               "threshold_source": report["threshold_source"]})
    score_table(splits[report["threshold_source"]], datasets["calibration"], errors, threshold).to_csv(
        output / "calibration_scores.csv", index=False)
    test_errors = score(model, loaders["test"], device,
                        description="Test scoring", show_progress=show_progress)
    table = score_table(splits["test"], datasets["test"], test_errors, threshold)
    table.to_csv(output / "test_scores.csv", index=False)
    save_point_scores(splits["test"], datasets["test"], test_errors, threshold,
                      output / "test_point_scores.csv")
    truth = table["binary_label"].to_numpy(dtype=bool)
    prediction = table["is_anomaly"].to_numpy(dtype=bool)
    tp, fp = int((truth & prediction).sum()), int((~truth & prediction).sum())
    fn, tn = int((truth & ~prediction).sum()), int((~truth & ~prediction).sum())
    metrics = {"threshold": threshold, "test_windows": len(table),
               "test_mean_mse": float(test_errors.mean()), "true_positive": tp,
               "false_positive": fp, "false_negative": fn, "true_negative": tn,
               "precision": tp / (tp + fp) if tp + fp else None,
               "recall": tp / (tp + fn) if tp + fn else None,
               "normal_window_false_positive_rate": fp / (fp + tn) if fp + tn else None,
               "label_definition": "Window endpoint: NORMAL=0; BROKEN=1; RECOVERING=1."}
    save_json(output / "test_metrics.json", metrics)
    print(f"Saved trained model, threshold, and test scores to {output}. "
          f"Total elapsed: {tqdm.format_interval(perf_counter() - run_started)}.", flush=True)
    return metrics


def predict(checkpoint_path: str, data_path: str, output_path: str,
            device_name="auto", stride=1, batch_size=64, split="all"):
    if stride < 1 or batch_size < 1:
        raise ValueError("Stride and batch size must be positive.")
    device = select_device(device_name)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    if "threshold" not in checkpoint:
        raise ValueError("Checkpoint is not calibrated. Complete training before prediction.")
    config = Config(**checkpoint["config"])
    torch.set_num_threads(config.cpu_threads)
    processor = Preprocessor(**checkpoint["preprocessing"])
    frame = read_data(data_path, require_status=False)
    if split == "test":
        frame = split_data(frame, config.train_fraction, config.validation_fraction,
                           config.calibration_fraction)["test"]
    elif split != "all":
        raise ValueError("Split must be all or test.")
    dataset = WindowDataset(processor.transform(frame), frame["timestamp"],
                            config.window_size, stride,
                            expected_interval_seconds=config.expected_interval_seconds)
    model = LSTMAutoencoder(**checkpoint["architecture"]).to(device)
    model.load_state_dict(checkpoint["model_state"])
    errors = (score(model, DataLoader(dataset, batch_size=batch_size), device)
              if len(dataset) else np.empty(0))
    result = score_table(frame, dataset, errors, checkpoint["threshold"])
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False)
    point_output = output.with_name(output.stem + "_points.csv")
    points = save_point_scores(frame, dataset, errors, checkpoint["threshold"], point_output)
    print(f"Saved {len(result)} window scores to {output}; {len(points)} row scores to "
          f"{point_output}. Threshold: {checkpoint['threshold']:.6g}; "
          f"exceedances: {int(points.is_anomaly.sum())}. Plot: {point_output.with_suffix('.png')}",
          flush=True)
    return result


def recalibrate(checkpoint_path, data_path, output_dir, cooldown_hours=24.0,
                device_name="auto", batch_size=2048):
    """Recalibrate unchanged weights on the original CSV's calibration interval."""
    output = Path(output_dir)
    if output.resolve() == Path(checkpoint_path).resolve().parent or output.exists():
        raise FileExistsError("Choose a new recalibration output directory.")
    if batch_size < 1:
        raise ValueError("Batch size must be positive.")
    device = select_device(device_name)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    config = Config(**checkpoint["config"])
    config.calibration_cooldown_hours = cooldown_hours
    config.validate()
    torch.set_num_threads(config.cpu_threads)
    frame = read_data(data_path)
    splits = split_data(frame, config.train_fraction, config.validation_fraction,
                        config.calibration_fraction)
    source = "calibration" if "calibration" in splits else "validation"
    part = splits[source]
    offset = len(splits["train"]) + (len(splits["validation"]) if source == "calibration" else 0)
    mask = calibration_normal_mask(frame, cooldown_hours)[offset:offset + len(part)]
    processor = Preprocessor(**checkpoint["preprocessing"])
    dataset = WindowDataset(processor.transform(part), part.timestamp,
                            config.window_size, config.stride, mask,
                            config.expected_interval_seconds)
    model = LSTMAutoencoder(**checkpoint["architecture"]).to(device)
    model.load_state_dict(checkpoint["model_state"])
    errors = score(model, DataLoader(dataset, batch_size=batch_size), device)
    threshold = float(np.quantile(errors, config.threshold_quantile))
    metadata = {
        "source_checkpoint": str(Path(checkpoint_path).resolve()),
        "source_data": str(Path(data_path).resolve()),
        "previous_threshold": checkpoint.get("threshold"),
        "threshold_source": source,
        "mse_threshold": threshold, "quantile": config.threshold_quantile,
        "calibration_cooldown_hours": cooldown_hours,
        "normal_calibration_windows": len(errors),
        "ignored_calibration_windows": len(dataset.ignored_starts),
        "calibration_start": str(part.timestamp.iloc[0]),
        "calibration_end": str(part.timestamp.iloc[-1]),
        "assumption": "NORMAL labels during the post-recovery settling interval are "
                      "excluded from threshold fitting. Duration requires operational validation.",
    }
    checkpoint["threshold"] = threshold
    checkpoint["config"] = asdict(config)
    checkpoint["calibration"] = metadata
    output.mkdir(parents=True)
    torch.save(checkpoint, output / "best_model.pt")
    save_json(output / "threshold.json", metadata)
    score_table(part, dataset, errors, threshold).to_csv(output / "calibration_scores.csv", index=False)
    print(f"Recalibrated threshold: {threshold:.6f} from {len(errors)} windows. "
          f"Weights unchanged; saved to {output}.", flush=True)
    predict(str(output / "best_model.pt"), data_path, str(output / "test_inference.csv"),
            device_name, 1, batch_size, "test")
    return metadata


def main():
    parser = argparse.ArgumentParser(description="Pump LSTM reconstruction anomaly detector")
    commands = parser.add_subparsers(dest="command", required=True)
    training = commands.add_parser("train", help="Train or verify with --dry-run")
    training.add_argument("--config", default="config.json")
    training.add_argument("--data", help="Override training CSV")
    training.add_argument("--output", help="Override artifact directory")
    training.add_argument("--epochs", type=int)
    training.add_argument("--device", choices=["auto", "cpu", "cuda"])
    training.add_argument("--no-progress", action="store_true",
                          help="Hide live progress bars; keep epoch summaries")
    training.add_argument("--dry-run", action="store_true",
                          help="Validate full dataset and one training batch; save no model")
    prediction = commands.add_parser("predict", help="Score CSV with a calibrated checkpoint")
    prediction.add_argument("--checkpoint", default="artifacts/pump_lstm_60_20_20/best_model.pt")
    prediction.add_argument("--data", required=True)
    prediction.add_argument("--output", default="artifacts/predictions.csv")
    prediction.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    prediction.add_argument("--split", choices=["all", "test"], default="all",
                            help="Use test only with the original training CSV")
    prediction.add_argument("--stride", type=int, default=1)
    prediction.add_argument("--batch-size", type=int, default=64)
    calibration = commands.add_parser("recalibrate", help="Refit threshold with post-recovery exclusion")
    calibration.add_argument("--checkpoint", default="artifacts/pump_lstm_60_20_20/best_model.pt")
    calibration.add_argument("--data", required=True, help="Original training CSV with labels")
    calibration.add_argument("--output", required=True, help="New artifact directory")
    calibration.add_argument("--cooldown-hours", type=float, default=24.0)
    calibration.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    calibration.add_argument("--batch-size", type=int, default=2048)
    args = parser.parse_args()
    try:
        if args.command == "train":
            config = Config(**json.loads(Path(args.config).read_text(encoding="utf-8-sig")))
            for option, field in (("data", "data_path"), ("output", "output_dir"),
                                  ("epochs", "epochs"), ("device", "device")):
                if getattr(args, option) is not None:
                    setattr(config, field, getattr(args, option))
            train(config, dry_run=args.dry_run, show_progress=not args.no_progress)
        elif args.command == "recalibrate":
            recalibrate(args.checkpoint, args.data, args.output, args.cooldown_hours,
                        args.device, args.batch_size)
        else:
            predict(args.checkpoint, args.data, args.output, args.device,
                    args.stride, args.batch_size, args.split)
    except (ValueError, FileNotFoundError, FileExistsError) as exc:
        parser.error(str(exc))
