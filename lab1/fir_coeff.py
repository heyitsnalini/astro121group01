import ugradio
import numpy as np

test_sdr = ugradio.sdr.SDR(direct = True,sample_rate = 1e6)
print(test_sdr.get_fir_coeffs())