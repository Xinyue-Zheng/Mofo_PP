import numpy as np


def estimate_channel_periods(x: np.ndarray, default_period: int = 24) -> list:
    """Estimate the dominant period for each channel via FFT.  # MoFo++

    Algorithm per channel:
      1. Detrend: subtract a moving-average trend (window = min(default_period, T//4)).
      2. FFT the seasonal component, zero out DC.
      3. Find the dominant frequency index, convert to period = round(T / index).
      4. Clamp to [2, T//2]; fall back to default_period if index <= 0.

    Args:
        x: Input array of shape (T, C).
        default_period: Fallback period used when estimation fails.

    Returns:
        List of ints of length C, one estimated period per channel.
    """
    T, C = x.shape
    window = max(1, min(default_period, T // 4))
    periods = []

    for c in range(C):
        series = x[:, c].astype(float)

        # Detrend: subtract moving-average
        if window > 1:
            kernel = np.ones(window) / window
            # 'same' mode keeps length T; edge effects are minor for long series
            ma = np.convolve(series, kernel, mode='same')
            seasonal = series - ma
        else:
            seasonal = series

        # FFT on seasonal component
        fft_vals = np.fft.rfft(seasonal)
        fft_vals[0] = 0  # zero out DC component

        magnitudes = np.abs(fft_vals)
        dom_idx = int(np.argmax(magnitudes))

        if dom_idx <= 0:
            periods.append(default_period)
            continue

        # rfft index k → frequency k/T → period T/k
        period = round(T / dom_idx)
        period = max(2, min(int(period), T // 2))
        periods.append(period)

    return periods
