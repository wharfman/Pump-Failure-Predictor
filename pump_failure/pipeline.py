"""Train, calibrate, and score the pump LSTM autoencoder."""
import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import random

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from .data import Preprocessor, WindowDataset, read_data, split_data
from .model import LSTMAutoencoder, reconstruction_error


@dataclass
class Config:
    data_path: str = "dataset/sensor.csv"
    output_dir: str = "artifacts/pump_lstm"
    window_size: int = 150
    stride: int = 1
    expected_interval_seconds: int = 60
    train_fraction: float = 0.60
    validation_fraction: float = 0.15
    calibration_fraction: float = 0.10
    hidden_size: int = 64
    latent_size: int = 32
    dropout: float = 0.1
    batch_size: int = 64
    epochs: int = 50
    learning_rate: float = 0.001
    weight_decay: float = 0.00001
    patience: int = 7
    gradient_clip: float = 1.0
    threshold_quantile: float = 0.99
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
        for name in ("learning_rate", "gradient_clip", "weight_decay"):
            value = getattr(self, name)
            if not np.isfinite(value) or value < 0 or (name != "weight_decay" and value == 0):
                raise ValueError(f"Invalid {name}.")
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
    for name, part in splits.items():
        normal = part["machine_status"].eq("NORMAL").to_numpy(dtype=bool)
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
            "statuses": {str(k): int(v) for k, v in part["machine_status"].value_counts().items()},
            "missing_fraction": {k: float(v) for k, v in part[processor.features].isna().mean().items()},
        }
    return splits, datasets, processor, report


def make_loader(dataset, config, shuffle=False):
    return DataLoader(dataset, batch_size=config.batch_size, shuffle=shuffle,
                      num_workers=config.num_workers,
                      pin_memory=select_device(config.device).type == "cuda")


def run_epoch(model, loader, device, optimizer=None, gradient_clip=1.0):
    model.train(optimizer is not None)
    total, count = 0.0, 0
    with torch.set_grad_enabled(optimizer is not None):
        for x in loader:
            x = x.to(device, non_blocking=True)
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
            loss = reconstruction_error(x, model(x)).mean()
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite reconstruction loss.")
            if optimizer is not None:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip,
                                               error_if_nonfinite=True)
                optimizer.step()
            total += loss.item() * len(x)
            count += len(x)
    return total / count


@torch.inference_mode()
def score(model, loader, device):
    model.eval()
    errors = []
    for x in loader:
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
        abnormal = ~frame["machine_status"].eq("NORMAL").fillna(False).to_numpy(dtype=bool)
        cumulative = np.r_[0, np.cumsum(abnormal)]
        result["contains_abnormal_status"] = (cumulative[ends + 1] - cumulative[starts]) > 0
    return result


def train(config: Config, dry_run=False):
    config.validate()
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
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate,
                                  weight_decay=config.weight_decay)
    output = Path(config.output_dir)
    if dry_run:
        x = next(iter(loaders["train"])).to(device)
        reconstructed = model(x)
        loss = reconstruction_error(x, reconstructed).mean()
        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite dry-run loss.")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip,
                                       error_if_nonfinite=True)
        optimizer.step()
        output.mkdir(parents=True, exist_ok=True)
        report["dry_run"] = {"input_shape": list(x.shape), "output_shape": list(reconstructed.shape),
                             "mse": float(loss.item()), "device": str(device)}
        save_json(output / "dry_run_report.json", report)
        print("Dry run passed: forward, backward, optimizer step. No checkpoint saved.", flush=True)
        return report

    output.mkdir(parents=True, exist_ok=True)
    if (output / "best_model.pt").exists():
        raise FileExistsError("Output already contains best_model.pt; choose a new --output directory.")
    save_json(output / "config.json", asdict(config))
    save_json(output / "data_report.json", report)
    save_json(output / "preprocessing.json", processor.to_dict())
    history, best, stale = [], float("inf"), 0
    for epoch in range(1, config.epochs + 1):
        train_loss = run_epoch(model, loaders["train"], device, optimizer, config.gradient_clip)
        val_loss = run_epoch(model, loaders["validation"], device)
        history.append({"epoch": epoch, "train_mse": train_loss, "validation_mse": val_loss})
        save_json(output / "history.json", history)
        print(f"Epoch {epoch:03d}: train={train_loss:.6f} validation={val_loss:.6f}", flush=True)
        if val_loss < best:
            best, stale = val_loss, 0
            torch.save({"format_version": 1, "model_state": model.state_dict(),
                        "architecture": architecture, "config": asdict(config),
                        "preprocessing": processor.to_dict(), "epoch": epoch,
                        "validation_mse": best}, output / "best_model.pt")
        else:
            stale += 1
            if stale >= config.patience:
                print("Early stopping.", flush=True)
                break
    checkpoint = torch.load(output / "best_model.pt", map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state"])
    errors = score(model, loaders["calibration"], device)
    threshold = float(np.quantile(errors, config.threshold_quantile))
    checkpoint["threshold"] = threshold
    torch.save(checkpoint, output / "best_model.pt")
    save_json(output / "threshold.json",
              {"mse_threshold": threshold, "quantile": config.threshold_quantile,
               "normal_calibration_windows": len(errors)})
    score_table(splits["calibration"], datasets["calibration"], errors, threshold).to_csv(
        output / "calibration_scores.csv", index=False)
    test_errors = score(model, loaders["test"], device)
    table = score_table(splits["test"], datasets["test"], test_errors, threshold)
    table.to_csv(output / "test_scores.csv", index=False)
    truth = table["contains_abnormal_status"].to_numpy(dtype=bool)
    prediction = table["is_anomaly"].to_numpy(dtype=bool)
    tp, fp = int((truth & prediction).sum()), int((~truth & prediction).sum())
    fn, tn = int((truth & ~prediction).sum()), int((~truth & ~prediction).sum())
    metrics = {"threshold": threshold, "test_windows": len(table),
               "test_mean_mse": float(test_errors.mean()), "true_positive": tp,
               "false_positive": fp, "false_negative": fn, "true_negative": tn,
               "precision": tp / (tp + fp) if tp + fp else None,
               "recall": tp / (tp + fn) if tp + fn else None,
               "normal_window_false_positive_rate": fp / (fp + tn) if fp + tn else None,
               "label_definition": "Any non-NORMAL row inside the window; not future failure."}
    save_json(output / "test_metrics.json", metrics)
    print(f"Saved trained model, threshold, and test scores to {output}", flush=True)
    return metrics


def predict(checkpoint_path: str, data_path: str, output_path: str,
            device_name="auto", stride=1, batch_size=64):
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
    dataset = WindowDataset(processor.transform(frame), frame["timestamp"],
                            config.window_size, stride,
                            expected_interval_seconds=config.expected_interval_seconds)
    model = LSTMAutoencoder(**checkpoint["architecture"]).to(device)
    model.load_state_dict(checkpoint["model_state"])
    errors = score(model, DataLoader(dataset, batch_size=batch_size), device)
    result = score_table(frame, dataset, errors, checkpoint["threshold"])
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False)
    print(f"Saved {len(result)} window scores to {output}", flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description="Pump LSTM reconstruction anomaly detector")
    commands = parser.add_subparsers(dest="command", required=True)
    training = commands.add_parser("train", help="Train or verify with --dry-run")
    training.add_argument("--config", default="config.json")
    training.add_argument("--data", help="Override training CSV")
    training.add_argument("--output", help="Override artifact directory")
    training.add_argument("--epochs", type=int)
    training.add_argument("--device", choices=["auto", "cpu", "cuda"])
    training.add_argument("--dry-run", action="store_true",
                          help="Validate full dataset and one training batch; save no model")
    prediction = commands.add_parser("predict", help="Score CSV with a calibrated checkpoint")
    prediction.add_argument("--checkpoint", default="artifacts/pump_lstm/best_model.pt")
    prediction.add_argument("--data", required=True)
    prediction.add_argument("--output", default="artifacts/predictions.csv")
    prediction.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    prediction.add_argument("--stride", type=int, default=1)
    prediction.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()
    try:
        if args.command == "train":
            config = Config(**json.loads(Path(args.config).read_text(encoding="utf-8-sig")))
            for option, field in (("data", "data_path"), ("output", "output_dir"),
                                  ("epochs", "epochs"), ("device", "device")):
                if getattr(args, option) is not None:
                    setattr(config, field, getattr(args, option))
            train(config, dry_run=args.dry_run)
        else:
            predict(args.checkpoint, args.data, args.output, args.device,
                    args.stride, args.batch_size)
    except (ValueError, FileNotFoundError, FileExistsError) as exc:
        parser.error(str(exc))
