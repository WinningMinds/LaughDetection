# ==================================================================================================
# Copyright 2017 The TensorFlow Authors All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==================================================================================================
"""
MEL Features
============

Defines routines to compute mel spectrogram features from audio waveform.
"""
import numpy as np


def frame(data: np.ndarray, window_length: int, step_size: int) -> np.ndarray:
    """Convert array into a sequence of successive, possibly overlapping, frames.

    A n-dimensional array of shape (num_samples, ...) is converted into a
    (n + 1)-D array of shape (num_frames, window_length, ...), where each
    frame start `step_size` points after the preceding one.

    Parameters
    ----------
    data
        The array containing the data.
    window_length
        Number of samples in each frame.
    step_size
        Advance (in samples) between each window.

    Returns
    -------
    :class:`~numpy.ndarray`
        The framed array.
    """
    num_samples = data.shape[0]
    num_frames = 1 + int(np.floor((num_samples - window_length) / step_size))
    shape = (num_frames, window_length) + data.shape[1:]
    strides = (data.strides[0] * step_size,) + data.strides
    return np.lib.stride_tricks.as_strided(data, shape=shape, strides=strides)


def periodic_hann(window_length: int) -> np.ndarray:
    """Caclulate "periodic" Hann window.

    The classic Hann window is defined as a raised cosine that starts and ends on zero, and
    where every value apperas twice, except the middle point for an odd-length window. Matlab
    calls this a "symmetric" window and :func:`~numpy.hanning` returns it. However, for
    Fournier analysis, this actually represents just over one cycle of period N-1 cosine, and
    thus is not compactly expressed on a length-N Fourier basis. Instead, it's better to use a
    raised cosine that ends just before the final zero value - i.e. a complete cycle of a
    period-N cosine. Matlab calls this a "periodic" window. This function calculates it.

    Parameters
    ----------
    window_length
        The number of points in the returned window.

    Returns
    -------
    :class:`~numpy.ndarray`
        The array containing the periodic Hann window.
    """
    return 0.5 - (0.5 * np.cos(2 * np.pi / window_length * np.arange(window_length)))


def sftf_magnitude(
    signal: np.ndarray, fft_length: float, window_length: int, step_size: int
) -> np.ndarray:
    """Calculate the short-time Fourier transform magnitude.
    
    Parameters
    ----------
    signal
        The array with the input time-domain signal.
    fft_length
        Size of the FFT to apply.
    window_length
        Length of each block of sample to pass to FFT.
    step_size
        Advance (in samples) between each frame passed to FFT.

    Returns
    -------
    :class:`~numpy.ndarray`
        Each row of the array contains the magnitudes of the
        `fft_length / 2 + 1` unique values of the FFT for the
        corresponding frame of input samples.
    """
    frames = frame(signal, window_length, step_size)
    window = periodic_hann(window_length)
    windowed_frames = frames * window
    return np.abs(np.fft.rfft(windowed_frames, int(fft_length)))


_MEL_BREAK_FREQUENCY_HERTZ = 700.0
_MEL_HIGH_FREQUENCY_Q = 1127.0


def hertz_to_mel(frequencies_hertz):
    """Convert frequencies to mel scale using HTK formula.

    Parameters
    ----------
    frequencies_hertz
        Scalar or array-like of frequencies in Hertz.

    Returns
    -------
    :class:`~numpy.ScalarType` or :class:`~numpy.ndarray`
        Corresponding values on the mel scale.
    """
    return _MEL_HIGH_FREQUENCY_Q * np.log(1.0 + (frequencies_hertz / _MEL_BREAK_FREQUENCY_HERTZ))


def spectrogram_to_mel_matrix(  # pylint: disable=too-many-locals
    num_mel_bands: int = 20,
    num_spectrogram_bins: int = 129,
    sample_rate: int = 8000,
    lower_edge_hertz: float = 125.0,
    upper_edge_hertz: float = 3800.0,
) -> np.ndarray:
    """Calculate matrix that can post-multiply spectrogram rows to make mel.

    Returns a np.array matrix A that can be used to post-multiply a matrix S of
    spectrogram values (STFT magnitudes) arranged as frames x bins to generate a
    "mel spectrogram" M of frames x num_mel_bins. M = S A.

    The classic HTK algorithm exploits the complementarity of adjacent mel bands
    to multiply each FFT bin by only one mel weight, then add it, with positive
    and negative signs, to the two adjacent mel bands to which that bin
    contributes.  Here, by expressing this operation as a matrix multiply, we go
    from num_fft multiplies per frame (plus around 2*num_fft adds) to around
    num_fft^2 multiplies and adds.  However, because these are all presumably
    accomplished in a single call to np.dot(), it's not clear which approach is
    faster in Python.  The matrix multiplication has the attraction of being more
    general and flexible, and much easier to read.

    Parameters
    ----------
    num_mel_bands
        How many bands in the resulting mel spectrum. This is the number of columns
        in the output matrix.
    num_spectrogram_bins
        How many bins there are in the source spectrogram data, which is understood
        to be `fft_size / 2 + 1`, i.e. the spectrogram only contains the nonredundant
        FFT bins.
    sample_rate
        Samples per second of the audio at the input to the spectrogram. We need this
        to figure out the actual frequencies for each spectrogram bin, which dictates
        how they are mapped into mel.
    lower_edge_hertz
        Lower bound on the frequencies to be included in the mel spectrum. This
        corresponds to the lower edge of the lowest triangular band.
    upper_edge_hertz
        The desired top edge of the highest frequency band.

    Returns
    -------
    :class:`~numpy.ndarray`
        The mel features.

    Raises
    ------
    :class:`ValueError`
        If frequency edges are incorrectly ordered out of range.
    """
    nyquist = sample_rate / 2
    if lower_edge_hertz < 0.0:
        raise ValueError(f"lower_edge_hertz {lower_edge_hertz:.1f} must be >= 0")
    if lower_edge_hertz > upper_edge_hertz:
        raise ValueError(
            f"lower_edge_hertz {lower_edge_hertz:.1f} must be "
            f"< {upper_edge_hertz:.1f} upper_edge_hertz"
        )
    if upper_edge_hertz > nyquist:
        raise ValueError(
            f"upper_edge_hertz {upper_edge_hertz:.1f} is greater than Nyquist {nyquist:.1f}"
        )
    spectrogram_bins_hertz = np.linspace(0.0, nyquist, num_spectrogram_bins)
    spectrogram_bins_mel = hertz_to_mel(spectrogram_bins_hertz)
    band_edges_mel = np.linspace(
        hertz_to_mel(lower_edge_hertz), hertz_to_mel(upper_edge_hertz), num_mel_bands + 2
    )
    mel_weights_matrix = np.empty((num_spectrogram_bins, num_mel_bands))

    for i in range(num_mel_bands):
        lower_edge_mel, center_mel, upper_edge_mel = band_edges_mel[i : i + 3]
        lower_slope = (spectrogram_bins_mel - lower_edge_mel) / (center_mel - lower_edge_mel)
        upper_slope = (upper_edge_mel - spectrogram_bins_mel) / (upper_edge_mel - center_mel)
        mel_weights_matrix[:, i] = np.maximum(0.0, np.minimum(lower_slope, upper_slope))

    mel_weights_matrix[0, :] = 0.0
    return mel_weights_matrix


def log_mel_spectrogram(
    data: np.ndarray,
    sample_rate: int = 8000,
    log_offset: float = 0.0,
    window_length: float = 0.025,
    step_size: float = 0.010,
    **kwargs,
) -> np.ndarray:
    """Convert waveform to a log magnitude mel-frequency spectrogram.

    Parameters
    ----------
    data
        1D :class:`~numpy.ndarray` of waveform data.
    sample_rate
        The sampling rate of the data.
    log_offset
        Add this to values when taking log to avoid -inf.
    window_length
        Duration of each window (in seconds) to analyze.
    step_size
        Advance (in seconds) between successive analysis windows.
    **kwargs
        Additional arguments to pass to `spectrogram_to_mel_matrix`.

    Returns
    -------
    :class:`~numpy.ndarray`
        Array of `(num_frames, num_mel_bands)` consisting of log mel filterbank
        magnitudes for successive frames.
    """
    window_length_samples = int(round(sample_rate * window_length))
    step_size_samples = int(round(sample_rate * step_size))
    fft_length = 2 ** int(np.ceil(np.log(window_length_samples) / np.log(2.0)))
    spectrogram = sftf_magnitude(
        data,
        fft_length=fft_length,
        window_length=window_length_samples,
        step_size=step_size_samples,
    )
    mel_spectrogram = np.dot(
        spectrogram,
        spectrogram_to_mel_matrix(
            num_spectrogram_bins=spectrogram.shape[1], sample_rate=sample_rate, **kwargs
        ),
    )
    return np.log(mel_spectrogram + log_offset)
