"""
The program for shifting a signal spectrum to a given frequency
"""

import os
import numpy as np
import scipy.io.wavfile as sw
import scipy.signal as sgn
import matplotlib.pyplot as plt


# Parameters
W0 = 0.02
WINDOW_SIZE = 500
BAND_ORDER = 3
FILTERING_ORDER = 5

RECORDS_PATH = "records"
RECORD_NAME = "signal.wav"
PLOTS_PATH = "plots"


def shift_frequencies_band(band, w0, filter_order, filter_freq, filter_btype='highpass'):
    """
    Frequencies band shifting function
    """
    shifted_band = band * np.cos(2 * np.pi * w0 * np.arange(len(band)))

    freq = filter_freq + w0 * 2
    if freq < 1:
        b, a = sgn.iirfilter(filter_order, freq, btype=filter_btype)
        shifted_band = sgn.filtfilt(b, a, shifted_band)

    return shifted_band * 2


def shift_signal_spectrum(x, freqs, w0, window_size, band_order, filtering_order):
    """
    Signal spectrum shifting function using multiplication by sine and a comb filter
    """
    div = len(freqs) // window_size
    if len(freqs) % window_size:
        band_btypes = ['lowpass'] + ['bandpass'] * (div - 1) + ['highpass']
    else:
        band_btypes = ['lowpass'] + ['bandpass'] * (div - 2) + ['highpass']

    y = 0.0
    for i, band_btype in enumerate(band_btypes):
        if band_btype == 'lowpass':
            band_freq = freqs[(i + 1) * window_size - 1]
            filtering_freq = 0.0
        elif band_btype == 'highpass':
            band_freq = freqs[i * window_size]
            filtering_freq = band_freq
        else:
            band_freq = [freqs[i * window_size], freqs[(i + 1) * window_size - 1]]
            filtering_freq = band_freq[0]

        b, a = sgn.iirfilter(band_order, band_freq, btype=band_btype)
        band = sgn.filtfilt(b, a, x)

        shifted_band = shift_frequencies_band(band, w0, filtering_order, filtering_freq)
        y += shifted_band

    return y


def plot_signal_spectrum(x, sample_rate, title="", save_path=None):
    """
    Signal spectrum plotting function
    """
    fc = np.fft.fft(x)
    fc = np.abs(fc)[:len(fc) // 2 + 1]
    freqs = np.linspace(0, sample_rate / 2, len(fc))

    plt.figure()
    plt.plot(freqs, fc)

    plt.xlabel("Frequency")
    plt.ylabel("Amplitude")
    plt.title(title)

    if save_path is not None:
        plt.savefig(os.path.join(save_path, title.lower().replace(' ', '_') + ".png"))


def main():
    """
    Main called function
    """
    # Read a source signal
    sample_rate, x = sw.read(os.path.join(RECORDS_PATH, RECORD_NAME))
    # Plot a signal spectrum
    plot_signal_spectrum(x, sample_rate, title="Source signal spectrum", save_path=PLOTS_PATH)

    # Create a frequencies array
    freqs = np.linspace(0, 1, len(x))
    # Shift the signal spectrum
    y = shift_signal_spectrum(x, freqs, w0=W0, window_size=WINDOW_SIZE, band_order=BAND_ORDER,
                              filtering_order=FILTERING_ORDER)

    # Save the shifted signal
    sw.write(os.path.join(RECORDS_PATH, "shifted_" + RECORD_NAME), sample_rate,
             np.int16(y / np.abs(y).max() * np.iinfo(np.int16).max))
    # Plot a shifted signal spectrum
    plot_signal_spectrum(y, sample_rate, title="Shifted on " + str(int(W0 * sample_rate)) + " Hz signal spectrum",
                         save_path=PLOTS_PATH)
    plt.show()


if __name__ == "__main__":
    main()
