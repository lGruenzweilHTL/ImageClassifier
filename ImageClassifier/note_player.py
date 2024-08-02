import numpy as np
import sounddevice as sd


def generate_sine_wave(frequency, duration, sample_rate=44100):
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    wave = 0.5 * np.sin(2 * np.pi * frequency * t)
    return wave


def play_frequency(frequency, seconds):
    sample_rate = 44100
    wave = generate_sine_wave(frequency, seconds, sample_rate)
    sd.play(wave, sample_rate)
    sd.wait()
