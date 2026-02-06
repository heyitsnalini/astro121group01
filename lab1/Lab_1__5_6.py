import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits as pf
from matplotlib.colors import LogNorm
from numpy import log10
from scipy.optimize import curve_fit

class Frequency_Resolution():
    def sample_data(npz_name):
        npzfile = np.load(f'{npz_name}.npz')
        plt.style.use('classic')
        plt.plot(np.arange(len(npzfile['data'])), np.fft.fftshift(npzfile['data']), c='g')
        plt.xlim(0, 1000)
        plt.ylim(-30, 30)
        plt.xlabel('Sample Number')
        plt.ylabel('Amplitude (Unknown Units)')
        plt.title('Frequency Resolution Raw Data')
        plt.show()

    def resolution_fft_shift(npz_name):
        npzfile = np.load(f'{npz_name}.npz')
        vspec = np.fft.fft(npzfile['data'])
        plt.style.use('classic')
        plt.plot(np.arange(len(vspec))-1000, np.real(vspec), label='Real Component', c='r')
        plt.plot(np.arange(len(vspec))-1000, np.imag(vspec), label='Imaginary Component', c='b')
        plt.xlabel('Frequency (Unknown Units)')
        plt.ylabel('Voltage (Unknown Units)')
        plt.xlim(-1000, 1000)
        plt.legend()
        plt.title('Real versus Imaginary components of the Voltage Spectra')
        plt.show()
    
    def fft_zoom(npz_name, x_lower, x_upper):
        npzfile = np.load(f'{npz_name}.npz')
        vspec = np.fft.fft(npzfile['data'])
        plt.style.use('classic')
        plt.plot(np.arange(len(vspec))-1000, np.real(vspec), label='Real Component', c='r')
        plt.plot(np.arange(len(vspec))-1000, np.imag(vspec), label='Imaginary Component', c='b')
        plt.xlabel('Frequency (Unknown Units)')
        plt.ylabel('Voltage (Unknown Units)')
        plt.xlim(x_lower, x_upper)
        plt.legend()
        plt.title('Real versus Imaginary components of the Voltage Spectra (Zoomed in)')
        plt.show()