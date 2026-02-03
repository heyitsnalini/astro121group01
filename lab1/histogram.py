# import ugradio
import numpy as np
import matplotlib.pyplot as plt


npzfile = np.load("a2.4e6_noise.npz")

# print(npzfile['data'])
test_data = npzfile['data']




plt.hist(test_data, bins = 100)

plt.show()
