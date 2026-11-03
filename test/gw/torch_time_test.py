"""Check Torch time conversions independently of the default device."""

import numpy as np
import pytest


@pytest.mark.array_backend
@pytest.mark.parametrize("device", ["cpu", "meta", "cuda"])
@pytest.mark.parametrize("scalar", [False, True])
def test_torch_time_device(monkeypatch, device, scalar):
    """Keep the leap-second table with the input, including after import."""
    torch = pytest.importorskip("torch")
    from bilby.gw.compat import torch as torch_time
    from bilby.gw.time import greenwich_mean_sidereal_time, n_leap_seconds

    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    monkeypatch.setattr("bilby.compat.utils.BILBY_ARRAY_API", True)
    table = torch_time.LEAP_SECONDS
    assert table.device.type == "cpu"
    dates = np.array([0., 1119744015., 1119744016., 1119744017., 1126259641.25])
    if scalar:
        dates = dates[-1]
    expected_leaps = n_leap_seconds(dates)
    expected_gmst = greenwich_mean_sidereal_time(dates)
    date = torch.as_tensor(dates, dtype=torch.float64, device=device)
    # A meta input exposes CPU table mismatches without requiring a GPU.
    # Conversely, a meta default must not move CPU inputs off their device.
    with torch.device("meta" if device == "cpu" else "cpu"):
        for _ in range(2):
            leaps = n_leap_seconds(date)
            gmst = greenwich_mean_sidereal_time(date)
            assert leaps.device == date.device
            assert gmst.device == date.device
            assert leaps.shape == date.shape
            assert gmst.dtype == date.dtype
            if device != "meta":
                torch.testing.assert_close(
                    leaps, torch.as_tensor(expected_leaps, device=device),
                )
                torch.testing.assert_close(
                    gmst, torch.as_tensor(expected_gmst, device=device),
                    rtol=1e-12, atol=1e-10,
                )
    assert torch_time.LEAP_SECONDS is table
    assert table.device.type == "cpu"
