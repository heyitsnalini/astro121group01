import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

class Plotting:
    def create_signal_graphs(x, y):
        plt.scatter(x, y, label='Data')
        plt.title('Sampling and Nyquist Aliasing, Sample Count')
        plt.xlabel('Time (steps)')
        plt.ylabel('Voltage (mV)')
        plt.legend()
        plt.show()

    def sinusoidal_func(x, amplitude, frequency, phase, offset):
        return amplitude * np.sin(x * frequency + phase) + offset

    def curve_fit(x, y):
        guess_amplitude = np.std(y) * 2.**0.5
        guess_offset = np.mean(y)
        guess_frequency = 1.0
        guess_phase = 0.0
        initial_guesses = [guess_amplitude, guess_frequency, guess_phase, guess_offset]
        fitted_params, pcov = curve_fit(Plotting.sinusoidal_func, x, y, p0=initial_guesses)
        optimized_amplitude, optimized_frequency, optimized_phase, optimized_offset = fitted_params
        plt.scatter(x, y, label='Data')
        plt.plot(x, Plotting.sinusoidal_func(x, *fitted_params), label='Fitted Curve')
        plt.title('Sampling and Nyquist Aliasing, Sample Count')
        plt.xlabel('Time (steps)')
        plt.ylabel('Voltage (mV)')
        plt.legend()
        plt.show()

