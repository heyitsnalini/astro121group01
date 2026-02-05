import ugradio
import numpy as np
import matplotlib.pyplot as plt

RF = "15Mhz"
LO= "25Mhz"
fir_on = True
power = "0.22Vpp"
mixertype = "SSB"

sample_rates = np.arange(1e6,1e7,2e5)

for sampling in sample_rates:
    
    if fir_on:
        test_sdr = ugradio.sdr.SDR(direct = False,sample_rate = sampling, center_freq =25e6)
    else:
        test_sdr = ugradio.sdr.SDR(direct = False,sample_rate = sampling, fir_coeffs = 
                                   np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2047]))
    
    
    test_data = test_sdr.capture_data(nblocks = 2)[1]
    
    np.savez(f'7.3.3_real__{(sampling/1e6)}e6_{RF}_{LO}', data = test_data, RF_freq = RF, LO_freq = LO, mixer = mixertype, sampled_at = sampling, FIR = fir_on, power=power)
    test_sdr.__del__()
    
#7.1 - build DSB mixer - only need one sampling rate but could be good to have a range if we want to show anything with it
# need sampling rate to satisfy Nyquist - which FIR coeffs should we use? verify them @ start
# we should do some rudimentary plots to make sure it's ok

#7.2 don't need to capture new data - need to examine a power spectrum

#7.3 make SSB mixer - initially capture some test data and examine it to see how the real and imaginary parts show up?
#part 1 - use a short cable so there's no delay + take data
# use a long cable for one to create delay
# use built in mixer in the SDR by turning off direct sampling