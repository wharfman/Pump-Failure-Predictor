import numpy as np
import pandas as pd
import pytest
import torch

from pump_failure.data import (Preprocessor, WindowDataset, binary_status_labels, calibration_normal_mask,
                               read_data, split_data)
from pump_failure.model import LSTMAutoencoder, reconstruction_error, reconstruction_loss
from pump_failure.pipeline import Config, map_reconstruction_errors, predict, recalibrate, run_epoch, score, train


def frame(n=40):
    return pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=n, freq="min"),
        "sensor_00": np.arange(n, dtype=float),
        "sensor_01": np.ones(n),
        "sensor_15": np.full(n, np.nan),
        "machine_status": ["NORMAL"] * n,
    })


def test_model_shapes_gradients_and_mse():
    torch.set_num_threads(2)
    model = LSTMAutoencoder(3, hidden_size=8, latent_size=4)
    x = torch.randn(2, 150, 3)
    output = model(x)
    assert output.shape == x.shape
    reconstruction_error(x, output).mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    assert torch.equal(reconstruction_error(torch.zeros_like(x), torch.ones_like(x)),
                       torch.ones(2))
    assert model.encoder.num_layers == model.decoder.num_layers == 2


def test_preprocessor_uses_normal_training_only_and_roundtrips():
    data = frame(10)
    data.loc[9, ["sensor_00", "machine_status"]] = [1e9, "BROKEN"]
    processor = Preprocessor.fit(data)
    assert processor.dropped_features == ["sensor_15"]
    assert processor.means[0] == 4.0
    assert processor.scales[1] == 1.0
    future = frame(2)
    future.loc[0, "sensor_00"] = np.nan
    before = processor.to_dict()
    transformed = processor.transform(future)
    assert np.isfinite(transformed).all()
    assert transformed[0, 0] == 0
    assert processor.to_dict() == before
    np.testing.assert_array_equal(transformed, Preprocessor(**before).transform(future))
    with pytest.raises(ValueError, match="Missing required"):
        processor.transform(future.drop(columns=["sensor_00"]))


def test_windows_exclude_failures_and_gaps_without_joining_rows():
    data = frame(12)
    data.loc[4, "machine_status"] = "BROKEN"
    data.loc[8:, "timestamp"] += pd.Timedelta(minutes=5)
    ds = WindowDataset(np.zeros((12, 2), dtype=np.float32), data.timestamp,
                       window_size=3, normal_mask=data.machine_status.eq("NORMAL").to_numpy())
    assert ds.starts.tolist() == [0, 1, 5, 8, 9]
    assert ds[0].shape == (3, 2)
    strided = WindowDataset(np.zeros((12, 2)), data.timestamp, window_size=3, stride=3)
    assert strided.starts.tolist() == [0, 3, 9]
    short = WindowDataset(np.zeros((2, 2)), data.timestamp.iloc[:2], window_size=150)
    assert len(short) == 0


def test_splits_have_disjoint_rows():
    splits = split_data(frame(100), .6, .2)
    assert [len(s) for s in splits.values()] == [60, 20, 20]
    parts = list(splits.values())
    for a, b in zip(parts, parts[1:]):
        assert a.timestamp.iloc[-1] < b.timestamp.iloc[0]
    with pytest.raises(ValueError):
        split_data(frame(), .8, .2, .1)


def test_loader_rejects_unsorted_timestamps(tmp_path):
    data = frame()
    path = tmp_path / "bad.csv"
    data.iloc[::-1].to_csv(path, index=False)
    with pytest.raises(ValueError, match="strictly increasing"):
        read_data(str(path))


def test_train_calibrate_reload_and_predict(tmp_path, monkeypatch):
    optimizers = []
    original_adam = torch.optim.Adam

    def capture_adam(*args, **kwargs):
        optimizer = original_adam(*args, **kwargs)
        optimizers.append(optimizer)
        return optimizer

    monkeypatch.setattr(torch.optim, "Adam", capture_adam)
    # Full 150-row architecture with tiny hidden layers and a short synthetic dataset.
    data = frame(2400)
    data["sensor_00"] = np.sin(np.arange(len(data)) / 20)
    data.loc[2280:2282, "machine_status"] = "BROKEN"
    path = tmp_path / "sensor.csv"
    data.to_csv(path, index=False)
    config = Config(data_path=str(path), output_dir=str(tmp_path / "run"),
                    hidden_size=4, latent_size=2, batch_size=4, epochs=1,
                    stride=150, dropout=0, cpu_threads=2, device="cpu")
    metrics = train(config)
    import json
    report = json.loads((tmp_path / "run" / "data_report.json").read_text())
    assert list(report["splits"]) == ["train", "validation", "test"]
    assert report["threshold_source"] == "validation"
    for name, expected in (("train", 1440), ("validation", 480), ("test", 480)):
        labels = pd.read_csv(tmp_path / "run" / f"{name}_labels.csv")
        assert len(labels) == expected
        np.testing.assert_array_equal(labels.binary_label, binary_status_labels(labels.machine_status))
    assert np.isfinite(metrics["threshold"])
    checkpoint = tmp_path / "run" / "best_model.pt"
    stored = torch.load(checkpoint, weights_only=True)
    assert type(optimizers[0]) is original_adam
    assert stored["training_objective"] == {"loss": "huber", "delta": 1.0, "optimizer": "Adam"}
    assert "validation_huber" in stored
    assert stored["score_metric"] == "reconstruction_mse"
    assert stored["architecture"]["input_size"] == 2
    assert stored["config"]["window_size"] == 150
    # Match the held-out split: preprocessing must be identical after loading.
    held_out = split_data(data, .6, .2)["test"]
    test_path = tmp_path / "held_out.csv"
    held_out.drop(columns="machine_status").to_csv(test_path, index=False)
    predictions = predict(str(checkpoint), str(test_path), str(tmp_path / "scores.csv"),
                          device_name="cpu", stride=150, batch_size=4)
    points = pd.read_csv(tmp_path / "scores_points.csv")
    assert len(points) == len(held_out)
    assert (tmp_path / "scores_points.png").stat().st_size > 0
    assert (tmp_path / "run" / "test_point_scores.png").stat().st_size > 0
    exceedances = pd.read_csv(tmp_path / "scores_points_exceedances.csv")
    assert exceedances.row_index.tolist() == points.loc[points.is_anomaly, "row_index"].tolist()
    original = pd.read_csv(tmp_path / "run" / "test_scores.csv")
    np.testing.assert_allclose(predictions.reconstruction_mse, original.reconstruction_mse,
                               rtol=1e-6)
    assert "end_status" not in predictions
    assert predictions.is_anomaly.tolist() == original.is_anomaly.tolist()
    # Recalibration reads fixed preprocessing and calibration data, not test errors.
    changed = data.copy()
    changed.loc[1920:, "sensor_00"] = 1000.0
    changed_path = tmp_path / "changed_test.csv"
    changed.to_csv(changed_path, index=False)
    recalibrated = tmp_path / "recalibrated"
    metadata = recalibrate(str(checkpoint), str(changed_path), str(recalibrated),
                           cooldown_hours=0, device_name="cpu", batch_size=4)
    assert metadata["mse_threshold"] == pytest.approx(stored["threshold"], rel=1e-6)
    new = torch.load(recalibrated / "best_model.pt", weights_only=True)
    assert new["preprocessing"] == stored["preprocessing"]
    for key in stored["model_state"]:
        assert torch.equal(new["model_state"][key], stored["model_state"][key])
    with pytest.raises(FileExistsError):
        recalibrate(str(checkpoint), str(path), str(recalibrated), device_name="cpu")
    with pytest.raises(FileExistsError):
        train(config)


@pytest.mark.skipif(
    __import__("sys").platform != "win32" or torch.version.hip is None
    or not torch.cuda.is_available(), reason="Requires Windows ROCm GPU")
def test_windows_rocm_native_lstm_matches_cpu():
    torch.manual_seed(123)
    cpu_model = LSTMAutoencoder(3, hidden_size=8, latent_size=4, dropout=0)
    gpu_model = LSTMAutoencoder(3, hidden_size=8, latent_size=4, dropout=0).to("cuda")
    gpu_model.load_state_dict(cpu_model.state_dict())
    backend_before = torch.backends.cudnn.enabled
    backend_during_lstm = []
    for layer in (gpu_model.encoder, gpu_model.decoder):
        layer.register_forward_pre_hook(
            lambda module, args: backend_during_lstm.append(torch.backends.cudnn.enabled))
    x = torch.randn(2, 150, 3)
    cpu_output = cpu_model(x)
    gpu_output = gpu_model(x.to("cuda"))
    assert backend_during_lstm == [False, False]
    assert torch.backends.cudnn.enabled == backend_before
    torch.testing.assert_close(gpu_output.cpu(), cpu_output, rtol=1e-4, atol=1e-5)
    cpu_output.square().mean().backward()
    gpu_output.square().mean().backward()
    for cpu_param, gpu_param in zip(cpu_model.parameters(), gpu_model.parameters()):
        assert gpu_param.grad is not None and torch.isfinite(gpu_param.grad).all()
        torch.testing.assert_close(gpu_param.grad.cpu(), cpu_param.grad, rtol=1e-3, atol=1e-5)
    gpu_model.eval()
    with torch.inference_mode():
        assert torch.isfinite(gpu_model(x.to("cuda"))).all()


def test_point_mapping_150_window_gaps_and_threshold():
    data = frame(455)
    data.loc[152:, "timestamp"] += pd.Timedelta(minutes=5)
    ds = WindowDataset(np.zeros((len(data), 2)), data.timestamp)
    assert ds.starts[:3].tolist() == [0, 1, 2]
    assert ds.ignored_starts.tolist() == list(range(3, 152))
    assert sorted(np.r_[ds.starts, ds.ignored_starts]) == ds.candidate_starts.tolist()
    errors = np.full(len(ds), 2.0)
    errors[:3] = [0.5, 1.0, 1.5]
    points = map_reconstruction_errors(data, ds, errors, 1.0)
    assert len(points) == len(data)
    assert (points.reconstruction_mse.iloc[:149] == 0).all()
    assert points.reconstruction_mse.iloc[149:152].tolist() == [0.5, 1.0, 1.5]
    assert points.is_anomaly.iloc[149:152].tolist() == [False, False, True]
    assert (points.score_status.iloc[152:301] == "ignored_window").all()
    assert (points.reconstruction_mse.iloc[152:301] == 0).all()
    assert not points.is_anomaly.iloc[152:301].any()
    assert points.reconstruction_mse.iloc[301] == 2.0
    np.testing.assert_array_equal(points.timestamp, data.timestamp)
    with pytest.raises(ValueError, match="one finite"):
        map_reconstruction_errors(data, ds, errors[:-1], 1.0)


def test_point_mapping_stride_and_no_valid_windows():
    data = frame(155)
    ds = WindowDataset(np.zeros((len(data), 2)), data.timestamp, stride=3)
    points = map_reconstruction_errors(data, ds, [0.0, 2.0], 1.0)
    assert points.index[points.is_scored].tolist() == [149, 152]
    assert points.score_status.iloc[150] == "not_selected"
    short = data.iloc[:20]
    ds = WindowDataset(np.zeros((len(short), 2)), short.timestamp)
    points = map_reconstruction_errors(short, ds, [], 1.0)
    assert not points.is_scored.any()
    assert (points.reconstruction_mse == 0).all()
    ds = WindowDataset(np.zeros((len(data), 2)), data.timestamp,
                       normal_mask=np.zeros(len(data), dtype=bool))
    points = map_reconstruction_errors(data, ds, [], 1.0)
    assert (points.score_status.iloc[149:] == "ignored_window").all()


def test_calibration_cooldown_preserves_original_window_positions():
    data = frame(400)
    data.loc[20:29, "machine_status"] = "RECOVERING"
    mask = calibration_normal_mask(data, cooldown_hours=1)
    assert mask[:20].all()
    assert not mask[20:90].any()
    assert mask[90:].all()
    # The split starts during cooldown: retain earlier recovery context.
    part = data.iloc[60:].reset_index(drop=True)
    ds = WindowDataset(np.zeros((len(part), 2)), part.timestamp,
                       window_size=150, normal_mask=mask[60:])
    assert ds.starts[0] == 30
    assert ds.ignored_starts.tolist() == list(range(30))
    np.testing.assert_array_equal(calibration_normal_mask(data, 0),
                                  data.machine_status.eq("NORMAL").to_numpy())
    for hours in [-1, float("nan"), float("inf")]:
        with pytest.raises(ValueError):
            calibration_normal_mask(data, hours)


def test_huber_loss_values_gradients_and_validation_epoch():
    target = torch.zeros(1, 2, 1)
    prediction = torch.tensor([[[0.5], [3.0]]], requires_grad=True)
    loss = reconstruction_loss(target, prediction)
    assert loss.item() == pytest.approx((0.125 + 2.5) / 2)
    loss.backward()
    torch.testing.assert_close(prediction.grad, torch.tensor([[[0.25], [0.5]]]))

    class ZeroModel(torch.nn.Module):
        def forward(self, x):
            return torch.zeros_like(x)

    loader = torch.utils.data.DataLoader(prediction.detach(), batch_size=1)
    assert run_epoch(ZeroModel(), loader, torch.device("cpu")) == pytest.approx(loss.item())
    assert run_epoch(ZeroModel(), loader, torch.device("cpu"), huber_delta=2.0) == pytest.approx(2.0625)


def test_inference_only_feeds_valid_windows_and_maps_original_indexes():
    data = frame(455)
    data.loc[152:, "timestamp"] += pd.Timedelta(minutes=5)
    # Row identity in input lets us inspect every window actually fed to the model.
    values = np.arange(len(data), dtype=np.float32)[:, None]
    ds = WindowDataset(values, data.timestamp, window_size=150)

    class RecordingModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.seen = []

        def forward(self, x):
            self.seen.extend(x[:, 0, 0].to(torch.int64).tolist())
            assert x.shape[1] == 150
            assert torch.all(x[:, -1, 0] - x[:, 0, 0] == 149)
            return x + 1

    model = RecordingModel()
    errors = score(model, torch.utils.data.DataLoader(ds, batch_size=7, shuffle=False),
                   torch.device("cpu"))
    assert model.seen == ds.starts.tolist()
    assert not set(model.seen).intersection(ds.ignored_starts.tolist())
    assert sorted(model.seen + ds.ignored_starts.tolist()) == ds.candidate_starts.tolist()
    points = map_reconstruction_errors(data, ds, errors, threshold=0.5)
    assert points.index[points.is_scored].tolist() == (ds.starts + 149).tolist()
    assert (points.reconstruction_mse.iloc[ds.starts + 149] == 1).all()
    assert (points.reconstruction_mse.iloc[ds.ignored_starts + 149] == 0).all()
    assert (points.reconstruction_mse.iloc[:149] == 0).all()


def test_three_categories_binary_labels_and_split_exports(tmp_path):
    data = frame(100)
    data.loc[60, "machine_status"] = "BROKEN"
    data.loc[61:65, "machine_status"] = "RECOVERING"
    data.loc[85, "machine_status"] = "BROKEN"
    data.loc[86:90, "machine_status"] = "RECOVERING"
    path = tmp_path / "labels.csv"
    data.to_csv(path, index=False)
    loaded = read_data(path)
    assert set(loaded.machine_status) == {"NORMAL", "BROKEN", "RECOVERING"}
    assert loaded.binary_label.iloc[60:67].tolist() == [1, 1, 1, 1, 1, 1, 0]
    parts = split_data(loaded)
    assert list(parts) == ["train", "validation", "test"]
    assert [len(p) for p in parts.values()] == [60, 20, 20]
    pd.testing.assert_frame_equal(pd.concat(parts.values(), ignore_index=True), loaded)
    ds = WindowDataset(np.zeros((100, 1)), loaded.timestamp, window_size=3)
    points = map_reconstruction_errors(loaded, ds, np.zeros(len(ds)), 1)
    assert points.binary_label.tolist() == loaded.binary_label.tolist()
    assert set(points.machine_status) == {"NORMAL", "BROKEN", "RECOVERING"}
    with pytest.raises(ValueError):
        binary_status_labels(pd.Series(["UNKNOWN"]))


def test_legacy_checkpoint_split_is_reproducible():
    assert [len(p) for p in split_data(frame(100), .6, .15, .1).values()] == [60, 15, 10, 15]
