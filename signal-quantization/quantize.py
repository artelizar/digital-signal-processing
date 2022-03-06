"""
The program for quantizing various signals and calculating their signal-to-noise ratio (SNR)
"""

import os
import enum
import math
import numpy as np
import scipy.io.wavfile as sw
import matplotlib.pyplot as plt


# Parameters
NUM_BITS = 10
THEORETICAL_SNR = 6 * NUM_BITS - 7.2

TIME = 1
SAMPLE_RATE = 1000
FS = 10

RECORDS_PATH = "records"
VOICE_RECORD_NAME = "voice.wav"
PLOTS_PATH = "plots"


class SignalType(enum.Enum):
    """
    Signal type enum class
    """
    SINUSOIDAL = 0
    UNIFORM_NOISE = 1
    VOICE = 2


def create_signal(signal_type, **kwargs):
    """
    Desired signal creating function
    """
    if signal_type == SignalType.SINUSOIDAL:
        t = np.linspace(0, kwargs['time'], kwargs['time'] * kwargs['sample_rate'])
        sn = np.sin(2 * np.pi * kwargs['fs'] * t)

        return sn
    elif signal_type == SignalType.UNIFORM_NOISE:
        un = np.random.uniform(-1, 1, kwargs['time'] * kwargs['sample_rate'])

        return un
    else:
        sample_rate, v = sw.read(kwargs['path'])
        v = v[:sample_rate * min(kwargs['time'], len(v) // sample_rate)]

        return v


def quantizate(x, num_bits):
    """
    Signal midrise uniform quantizing function
    """
    x_mn, x_mx = x.min(), x.max()
    h = (x_mx - x_mn) / 2 ** num_bits
    q = np.maximum(x_mn, np.minimum(x_mx, (np.floor(x / h) + 0.5) * h))
    e = x - q

    return q, e


def signal_to_noise_ratio(x, e):
    """
    Signal-to-noise ratio (SNR) implementation
    """
    return 10 * math.log10(np.var(x) / np.var(e))


def plot_hists(x, e, title="", save_path=None):
    """
    Signals and quantization errors' histograms plotting and saving function
    """
    fig, axs = plt.subplots(2, figsize=(10, 10))
    fig.suptitle(title)

    axs[0].hist(np.arange(len(x)), bins=len(x), weights=x)
    axs[0].set_title("Signal")
    axs[1].hist(np.arange(len(e)), bins=len(e), weights=e)
    axs[1].set_title("Quantization errors")

    if save_path is not None:
        plt.savefig(os.path.join(save_path, title.lower().replace(' ', '_') + ".png"))
    plt.show()


def process_signal(x, num_bits, title, save_path):
    """
    Selected signal processing function
    """
    # Quantizate the signal, calculate and print SNR
    _, e_x = quantizate(x, num_bits)
    x_snr = signal_to_noise_ratio(x, e_x)
    print(title + " noise signal's SNR:", round(x_snr, 1))

    # Plot and save signal's and quantization errors' histograms for the signal
    plot_hists(x, e_x, title, save_path)


def main():
    """
    Main called function
    """
    # Print theoretical SNR
    print("Theoretical SNR:", THEORETICAL_SNR)

    # Create and save a sinusoidal signal
    sn, sn_name = create_signal(SignalType.SINUSOIDAL, time=TIME, sample_rate=SAMPLE_RATE, fs=FS), "Sinusoidal"
    sw.write(os.path.join(RECORDS_PATH, sn_name.lower().replace(' ', '_') + ".wav"), SAMPLE_RATE,
             np.int16(sn * np.iinfo(np.int16).max))
    # Quantizate the sinusoidal signal and calculate SNR, plot and save signals and quantization errors' histograms
    process_signal(sn, num_bits=NUM_BITS, title=sn_name, save_path=PLOTS_PATH)

    # Create and save a uniform noise signal
    un, un_name = create_signal(SignalType.UNIFORM_NOISE, time=TIME, sample_rate=SAMPLE_RATE), "Uniform noise"
    sw.write(os.path.join(RECORDS_PATH, un_name.lower().replace(' ', '_') + ".wav"), SAMPLE_RATE,
             np.int16(un * np.iinfo(np.int16).max))
    # Quantizate the uniform noise signal and calculate SNR, plot and save signals and quantization errors' histograms
    process_signal(un, num_bits=NUM_BITS, title=un_name, save_path=PLOTS_PATH)

    # Read a voice signal
    v, v_name = create_signal(SignalType.VOICE, path=os.path.join(RECORDS_PATH, VOICE_RECORD_NAME), time=TIME), "Voice"
    # Quantizate the voice signal and calculate SNR, plot and save signals and quantization errors' histograms
    process_signal(v, num_bits=NUM_BITS, title=v_name, save_path=PLOTS_PATH)


if __name__ == "__main__":
    main()
