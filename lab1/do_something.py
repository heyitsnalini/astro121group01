import ugradio
import numpy as np
import matplotlib.pyplot as plt

# default_sdr = ugradio.sdr.SDR(direct = True,sample_rate = 1e6)
# aliasing_sdr = ugradio.sdr.SDR(direct = True, fir_coeffs = np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2047]))

freq = "noise"
fir_on = False
file_number=0
voltage = "noise"

sample_rates = np.arange(1e6, 3.2e6, 2e5)
for i in sample_rates:
    sampling = i
    
    if fir_on:
        test_sdr = ugradio.sdr.SDR(direct = True,sample_rate = sampling)
    else:
        test_sdr = ugradio.sdr.SDR(direct = True,sample_rate = sampling, fir_coeffs = 
                                   np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2047]))
    
    test_data = test_sdr.capture_data(nblocks = 2)[1]
    
    np.savez(f'a{(sampling/1e6)}e6_{freq}', data = test_data, frequency = freq, sampled_at = sampling, FIR = fir_on, vpp = voltage)
    test_sdr.__del__()
    
    
#note: the 1 mhz, 3.2 mhz, and 2.1 mhz files are all without overriding FIR coefficients and with a different signal generator than the rest, and with 100 kHz wave - didnt end up adding to npz file

# npzfile = np.load("2_1mhz.npz")

# print(npzfile['data'])
# test_data = npzfile['data']

# plt.plot(np.arange(len(test_data)), test_data)
# plt.xlim(0, 100)
# plt.ylim(-20, 20)
# plt.show()

# npzfile = np.load("3.2mhz.npz")

# print(npzfile['data'])
# test_data = npzfile['data']

# plt.plot(np.arange(len(test_data)), test_data)
# plt.xlim(0, 100)
# plt.ylim(-20, 20)
# plt.show()