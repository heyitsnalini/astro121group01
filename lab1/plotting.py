import ugradio
import numpy as np
import matplotlib.pyplot as plt


npzfile = np.load("1.4e6_1000kHz.npz")

# print(npzfile['data'])
test_data = npzfile['data']

npzfile2 = np.load("a1.4e6_1000kHz.npz")


test_data2 = npzfile2['data']

plt.plot(np.arange(len(test_data)), test_data)
plt.xlim(0, 70)
plt.ylim(-120, 120)
plt.show()

plt.plot(np.arange(len(test_data2)), test_data2)
plt.xlim(0, 70)
plt.ylim(-120, 120)
plt.show()