import numpy as np
import pandas as pd
import pytest
import torch

from pump_failure.data import Preprocessor, WindowDataset, read_data, split_data
from pump_failure.model import LSTMAutoencoder, reconstruction_error
from pump_failure.pipeline import Config, predict, train


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
    splits = split_data(frame(100), .6, .15, .1)
    assert [len(s) for s in splits.values()] == [60, 15, 10, 15]
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


def test_train_calibrate_reload_and_predict(tmp_path):
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
    assert np.isfinite(metrics["threshold"])
    checkpoint = tmp_path / "run" / "best_model.pt"
    stored = torch.load(checkpoint, weights_only=True)
    assert stored["architecture"]["input_size"] == 2
    assert stored["config"]["window_size"] == 150
    # Match the held-out split: preprocessing must be identical after loading.
    held_out = split_data(data, .6, .15, .1)["test"]
    test_path = tmp_path / "held_out.csv"
    held_out.drop(columns="machine_status").to_csv(test_path, index=False)
    predictions = predict(str(checkpoint), str(test_path), str(tmp_path / "scores.csv"),
                          device_name="cpu", stride=150, batch_size=4)
    original = pd.read_csv(tmp_path / "run" / "test_scores.csv")
    np.testing.assert_allclose(predictions.reconstruction_mse, original.reconstruction_mse,
                               rtol=1e-6)
    assert "end_status" not in predictions
    assert predictions.is_anomaly.tolist() == original.is_anomaly.tolist()
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
