"""
The program for demodulating a signal using its sequence of instantaneous frequencies
"""

import os
import numpy as np
import scipy.io.wavfile as sw
import scipy.signal as sgn
import matplotlib.pyplot as plt


# Parameters
WINDOW_SIZE = 1600
EPS = 5

ZEROS_FREQ = 3000
BASE_FREQ = 4000
ONES_FREQ = 5000

RECORD_PATH = "records/frequency_modulation.wav"
PLOTS_PATH = "plots"


def median_filter(input):
    """
    Median filtering function
    """
    return np.sort(input)[len(input) // 2]


def demodulate(instantaneous_freqs, window_size, zeros_freq, ones_freq, eps):
    """
    Signal demodulation function using a sequence of instantaneous frequencies
    """
    msg, prev_gap = [], False
    for i in range(0, len(instantaneous_freqs), window_size):
        freq = median_filter(instantaneous_freqs[i: i + window_size])

        if abs(freq - zeros_freq) < eps:
            msg.append(0)
            prev_gap = False
        elif abs(freq - ones_freq) < eps:
            msg.append(1)
            prev_gap = False
        else:
            if prev_gap:
                break
            else:
                prev_gap = True

    return msg


def main():
    """
    Main called function
    """
    # Read a source signal
    sample_rate, x = sw.read(os.path.join(*RECORD_PATH.split('/')))
    # Apply the Hilbert transform to the signal and get an analytical signal
    z = sgn.hilbert(x)

    # Plot and save a signal's sequence of instantaneous frequencies
    instantaneous_freqs = sample_rate / (2 * np.pi) * np.diff(np.unwrap(np.angle(z)))
    # Display the signal's sequence of instantaneous frequencies
    plt.plot(instantaneous_freqs)
    plt.xlabel("Time")
    plt.ylabel("Instantaneous frequencies")
    plt.title("Dependence of signal's instantaneous frequencies on time")
    plt.savefig(os.path.join(PLOTS_PATH, plt.gca().get_title().lower().replace(' ', '_') + ".png"))
    plt.show()

    # Demodulate the source signal
    msg = demodulate(instantaneous_freqs, window_size=WINDOW_SIZE, zeros_freq=ZEROS_FREQ, ones_freq=ONES_FREQ, eps=EPS)
    print("Message:", msg)


if __name__ == "__main__":
    main()
