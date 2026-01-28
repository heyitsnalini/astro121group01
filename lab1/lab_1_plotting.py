import numpy as np
import matplotlib.pyplot as plt

class Plotting:
    def create_signal_graphs(x, y):
        plt.scatter(x, y, label='Data')
        plt.title('Sampling and Nyquist Aliasing, Sample Count')
        plt.xlabel('Time (steps)')
        plt.ylabel('Voltage (mV)')
        plt.legend()
        plt.show()
