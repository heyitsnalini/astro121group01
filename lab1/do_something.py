import ugradio
import numpy as np
import matplotlib.pyplot as plt

#default_sdr = ugradio.sdr.SDR(direct = True,sample_rate = 1e6)
#aliasing_sdr = ugradio.sdr.SDR(direct = True, fir_coeffs = np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2047]))

# test_sdr = ugradio.sdr.SDR(direct = True,sample_rate = 3.2e6)
# test_data = test_sdr.capture_data(nblocks =2)[1]

# print(test_data)

# np.savez("3_2mhz.npz", data = test_data)

npzfile = np.load("2_1mhz.npz")

print(npzfile['data'])
test_data = npzfile['data']

plt.plot(np.arange(len(test_data)), test_data)
plt.xlim(0, 100)
plt.ylim(-20, 20)
plt.show()

npzfile = np.load("3.2mhz.npz")

print(npzfile['data'])
test_data = npzfile['data']

plt.plot(np.arange(len(test_data)), test_data)
plt.xlim(0, 100)
plt.ylim(-20, 20)
plt.show()