"""
The program for computing a voice signal's spectrogram and fundamental frequency
"""

import os
import numpy as np
import scipy.io.wavfile as sw
import matplotlib.pyplot as plt


# Parameters
WINDOW_SIZE = 512
BUFFER_SIZE = 100

SIGNAL_SECTION = slice(None, 100000)
VOICE_SECTION = slice(29000, 56000)
SILENCE_SECTION = slice(57000, 85000)

VOICE_VMIN = 0
SILENCE_VMIN = 3

RECORD_PATH = "records/voice.wav"
PLOTS_PATH = "plots"


def return_hann_window(size):
    """
    Hann window implementation for smoothing values
    """
    n = np.linspace(0, size - 1, num=size)
    hann_window = 0.5 * (1 - np.cos((2 * np.pi * n) / (size - 1)))

    return hann_window


def compute_spectrogram(x, window_size, buffer_size):
    """
    Spectrogram computing function using FFT and the Hann window
    """
    hann_window = return_hann_window(window_size)
    buffer = np.empty((buffer_size, window_size // 2 + 1))
    spectrogram = []

    steps_count = len(x) - window_size + 1
    for i in range(steps_count):
        fc = np.fft.fft(x[i:i + window_size] * hann_window)
        fc = np.abs(fc[:window_size // 2 + 1])

        buffer[i % buffer_size] = fc
        if i % buffer_size == buffer_size - 1:
            spectrogram.append(buffer.mean(axis=0))
    if (steps_count - 1) % buffer_size != buffer_size - 1:
        spectrogram.append(buffer[:(steps_count - 1) % buffer_size + 1].mean(axis=0))

    return np.column_stack(spectrogram)


def plot_graph(g, xlabel="", ylabel="", title="", save_path=None):
    """
    Graph plotting and saving function
    """
    plt.figure()
    plt.plot(g)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)

    if save_path is not None:
        plt.savefig(os.path.join(save_path, title.lower().replace(' ', '_') + ".png"))


def display_spectrogram(spectrogram, vmin, title="", save_path=None):
    """
    Spectrogram displaying and saving function
    """
    plt.figure()
    plt.imshow(spectrogram, cmap='gray', vmin=vmin)
    plt.gca().invert_yaxis()

    plt.xlabel("Time")
    plt.ylabel("Frequency")
    plt.title(title)

    if save_path is not None:
        plt.savefig(os.path.join(save_path, title.lower().replace(' ', '_') + ".png"))


def main():
    """
    Main called function
    """
    # Read and display a source signal
    sample_rate, x = sw.read(os.path.join(*RECORD_PATH.split('/')))
    plot_graph(x[SIGNAL_SECTION], xlabel="Time", ylabel="Frequency", title="Voice signal")

    # Compute and display the spectrogram of a signal section with a voice
    voice_spectrogram = compute_spectrogram(x[VOICE_SECTION], window_size=WINDOW_SIZE, buffer_size=BUFFER_SIZE)
    voice_spectrogram = np.log(voice_spectrogram)
    display_spectrogram(voice_spectrogram, vmin=VOICE_VMIN, title="Voice signal spectrogram", save_path=PLOTS_PATH)

    # Calculate and plot the fundamental frequency F0 of the signal section with the voice
    freqs = np.linspace(0, sample_rate / 2, len(voice_spectrogram))
    f0 = freqs[voice_spectrogram.argmax(axis=0)]
    plot_graph(f0, xlabel="Time", ylabel="F0", title="F0 dependence on time for a voice signal", save_path=PLOTS_PATH)

    # Compute and display the spectrogram of a signal section with a silence
    silence_spectrogram = compute_spectrogram(x[SILENCE_SECTION], window_size=WINDOW_SIZE, buffer_size=BUFFER_SIZE)
    silence_spectrogram = np.log(silence_spectrogram)
    display_spectrogram(
        silence_spectrogram,
        vmin=SILENCE_VMIN,
        title="Silence signal spectrogram",
        save_path=PLOTS_PATH
    )
    plt.show()


if __name__ == "__main__":
    main()
