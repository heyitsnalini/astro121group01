import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits as pf
from matplotlib.colors import LogNorm
from numpy import log10
from scipy.optimize import curve_fit
from scipy import signal

class PowerAndVoltage():

    def power_and_log_graph(npz_name):
        npzfile = np.load(f'{npz_name}.npz')
        vspec = np.fft.fft(npzfile['data'])
        vspec_new = np.fft.ifft((np.abs(vspec))**2)
        plt.style.use('classic')
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(10,12))
        ax2.plot(np.arange(len(vspec))-1000, (np.abs((vspec)))**2, label='Power Spectra', c='g')
        ax4.plot(np.arange(len(vspec))-1000, log10((np.abs((vspec)))**2), label='Log 10 of the Power Spectra', c='g')
        ax1.plot(np.arange(len(vspec))-1000, np.real(vspec), label='Real Component', c='r')
        ax1.plot(np.arange(len(vspec))-1000, np.imag(vspec), label='Imaginary Component', c='b')
        ax3.scatter(np.arange(len(vspec)), np.real(vspec), label='Real Component', c='r')
        ax3.scatter(np.arange(len(vspec)), np.imag(vspec), label='Imaginary Component', c='b')
        ax1.set_title('Voltage Spectra')
        ax2.set_title('Power Spectra')
        ax3.set_title('Voltage Spectra (zoomed-in)')
        ax4.set_title('Power Spectra (Logarithmic)')
        ax3.set_xlabel('Frequency (kHz)')
        ax4.set_xlabel('Frequency (kHz)')
        ax3.set_xlabel('Frequency (kHz)')
        ax4.set_xlabel('Frequency (kHz)')
        ax2.set_ylabel('Power (unknown units)')
        ax4.set_ylabel('Power (unknown units)')
        ax1.set_ylabel('Voltage (unknown units)')
        ax3.set_ylabel('Voltage (unknown units)')
        plt.xlim(-1000, 1000)
        ax1.set_xlim(-1000, 1000)
        ax2.set_xlim(-1000, 1000)
        ax3.set_xlim(1, 100)
        ax3.set_ylim(-100, 100)
        ax1.legend(fontsize='x-small')
        ax2.legend(fontsize='x-small')
        ax3.legend(fontsize='x-small')
        ax4.legend(fontsize='x-small')
        plt.show()
    
    def ifft_of_power(npz_name):
        npzfile = np.load(f'{npz_name}.npz')
        vspec = np.fft.fft(npzfile['data'])
        vspec_new = np.fft.ifft((np.abs(vspec))**2)
        plt.style.use('classic')
        plt.plot(np.arange(len(vspec)), np.real(vspec_new)/10000, label='Inverse Fourier Transform of Power Spectra', c='b')
        plt.plot(np.arange(len(npzfile['data'])), np.fft.fftshift(npzfile['data']), label='Original Sample', c='r')
        plt.xlim(0, 100)
        plt.ylim(-20, 20)
        plt.xlabel('Sample Number')
        plt.ylabel('Voltage (Unknown Units)')
        plt.title('Comparing the Inverse Fourier Transform of the Power Spectra to the Original Sample')
        plt.legend()
        plt.show()
    
    def correlation_theorem(npz_name):
        npzfile = np.load(f'{npz_name}.npz')
        vspec = np.fft.fft(npzfile['data'])
        vspec_new = np.fft.ifft((np.abs(vspec))**2)
        np_corr = np.correlate(npzfile['data'], vspec, mode='full')
        sp_corr = signal.correlate(npzfile['data'], vspec, mode='full')
        print(np_corr)
        print(sp_corr)

        plt.style.use('classic')
        fig, ((ax1, ax2)) = plt.subplots(2, 1, figsize=(10,10))
        ax1.plot(np.arange(len(np_corr)), np.real(np_corr), label='Real Component', c='r')
        ax2.plot(np.arange(len(sp_corr)), np.real(sp_corr), label='Real Component', c='r')
        ax1.plot(np.arange(len(np_corr)), np.imag(np_corr), label='Imaginary Component', c='b')
        ax2.plot(np.arange(len(sp_corr)), np.imag(sp_corr), label='Imaginary Component', c='b')
        ax1.legend(fontsize='x-small')
        ax2.legend(fontsize='x-small')
        ax1.set_title('Numpy Correlation')
        ax2.set_title('Scipy Correlation')
        ax1.set_xlabel('Sample Number')
        ax2.set_xlabel('Sample Number')
        plt.legend()
        plt.show()