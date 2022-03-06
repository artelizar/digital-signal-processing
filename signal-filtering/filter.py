"""
The program for designing a filter and filtering a signal
"""

import os
import numpy as np
import scipy.io.wavfile as sw
import scipy.signal as sgn
import matplotlib.pyplot as plt


# Parameters
FREQUENCIES = [15, 30, 50, 75, 90]
SAMPLE_RATE = 200
TIME = 20

# FIR filter specification
FIR_FILTER_ORDER = 1025
FIR_FILTER_FREQ = [0.0, 0.295, 0.295, 0.305, 0.305, 1.0]
FIR_FILTER_GAIN = [0, 0, 1, 1, 0, 0]
# IIR filter specification
IIR_FILTER_ORDER = 5
IIR_FILTER_FREQ = [0.745, 0.755]
IIR_FILTER_BTYPE = 'bandpass'

RECORDS_PATH = "records"
RECORD_NAME = "signal.wav"
PLOTS_PATH = "plots"


def create_signal(frequencies, sample_rate, time):
    """
    Generating a signal as a sum of 5 sinusoids with different frequencies
    """
    x = 0
    t = np.linspace(0, time, time * sample_rate)
    for freq in frequencies:
        x += np.sin(2 * np.pi * freq * t)

    return x


def plot_transfer_function(w, h, title="", save_path=None):
    """
    Signal transfer function plotting and saving function
    """
    plt.figure()
    plt.plot(w / np.pi, np.abs(h))
    plt.xlabel("Frequency")
    plt.ylabel("Amplitude")
    plt.title(title)

    if save_path is not None:
        plt.savefig(os.path.join(save_path, title.lower().replace(' ', '_') + ".png"))


def filter_signal(x, fir_filter_order, fir_filter_freq, fir_filter_gain, iir_filter_order, iir_filter_freq,
                  iir_filter_btype, plots_path):
    """
    Building a filter and filtering a signal
    """
    # Build a FIR filter according to the given specification
    fir_filter_b = sgn.firwin2(fir_filter_order, fir_filter_freq, fir_filter_gain)
    fir_filter_w, fir_filter_h = sgn.freqz(fir_filter_b)

    # Build an IIR filter according to the given specification
    iir_filter_b, iir_filter_a = sgn.iirfilter(iir_filter_order, iir_filter_freq, btype=iir_filter_btype)
    iir_filter_w, iir_filter_h = sgn.freqz(iir_filter_b, iir_filter_a)

    # Build a final filter with parallel connection of the FIR and IIR filters
    final_filter_w = fir_filter_w
    final_filter_h = fir_filter_h + iir_filter_h
    # Plot the transfer function of the final filter
    plot_transfer_function(final_filter_w, final_filter_h, title="Final filter transfer function", save_path=plots_path)

    # Filter the signal with phase shift minimization
    y = sgn.filtfilt(fir_filter_b, 1, x)
    y += sgn.filtfilt(iir_filter_b, iir_filter_a, x)

    return y


def plot_graph(args, xlabel="", ylabel="", title="", save_path=None):
    """
    Graph plotting and saving function
    """
    plt.figure()
    plt.plot(*args)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)

    if save_path is not None:
        plt.savefig(os.path.join(save_path, title.lower().replace(' ', '_') + ".png"))


def main():
    """
    Main called function
    """
    # Generate and save a source signal
    x = create_signal(frequencies=FREQUENCIES, sample_rate=SAMPLE_RATE, time=TIME)
    sw.write(os.path.join(RECORDS_PATH, RECORD_NAME), SAMPLE_RATE,
             np.int16(x / np.abs(x).max() * np.iinfo(np.int16).max))

    # Apply the Fast Fourier transform (FFT) to the signal
    x_fc = np.fft.fft(x[:SAMPLE_RATE])
    x_fc = np.abs(x_fc[:len(x_fc) // 2 + 1])
    # Create a frequencies array
    freqs = np.linspace(0, SAMPLE_RATE / 2, len(x_fc))
    # Plot the spectrum of the signal
    plot_graph([freqs, x_fc], xlabel="Frequency", ylabel="Amplitude",
               title="Source signal spectrum", save_path=PLOTS_PATH)

    # Filter the source signal
    y = filter_signal(x, fir_filter_order=FIR_FILTER_ORDER, fir_filter_freq=FIR_FILTER_FREQ,
                      fir_filter_gain=FIR_FILTER_GAIN, iir_filter_order=IIR_FILTER_ORDER,
                      iir_filter_freq=IIR_FILTER_FREQ, iir_filter_btype=IIR_FILTER_BTYPE, plots_path=PLOTS_PATH)

    # Apply the FFT to the filtered signal
    y_fc = np.fft.fft(y[:SAMPLE_RATE])
    y_fc = np.abs(y_fc[:len(y_fc) // 2 + 1])
    # Plot the spectrum of the filtered signal
    plot_graph([freqs, y_fc], xlabel="Frequency", ylabel="Amplitude",
               title="Filtered signal spectrum", save_path=PLOTS_PATH)

    # Save the filtered signal
    sw.write(os.path.join(RECORDS_PATH, "filtered_" + RECORD_NAME), SAMPLE_RATE,
             np.int16(y / np.abs(y).max() * np.iinfo(np.int16).max))

    # Display and save the source and filtered signals
    plt.figure()
    plt.plot(x[:SAMPLE_RATE], label="Source signal")
    plt.plot(y[:SAMPLE_RATE], label="Filtered signal")
    plt.xlabel("Time")
    plt.ylabel("Amplitude")
    plt.title("Source and filtered signals")
    plt.legend()
    plt.savefig(os.path.join(PLOTS_PATH, plt.gca().get_title().lower().replace(' ', '_') + ".png"))
    plt.show()


if __name__ == "__main__":
    main()
