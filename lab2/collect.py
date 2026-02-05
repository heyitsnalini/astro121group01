import ugradio
import numpy as np
import matplotlib.pyplot as plt

# example_sdr = ugradio.sdr.SDR(direct = False, sample_rate = [sample rate], center_freq = [LO of internal mixer])
# lab manual suggests setting LO off to side of 1420.405 MHz line
# example_sdr.capture_data(nsamples = [2048 default, can change it], nblocks = [????])



def collection(sample_rates, lab_section):

    for i in sample_rates:
    
        time = ugradio.timing.unix_time()
        location = np.array([ugradio.nch.lat, ugradio.nch.lon, ugradio.nch.alt])
        sampling = i
        
        our_sdr = ugradio.sdr.SDR(direct = False, sample_rate = sampling, center_freq = 1.420e9)
    
        out = our_sdr.capture_data(nblocks = 2)[1]
        data = np.zeros(len(out))
        
        print(len(data))
        
        for k in np.arange(len(data)):
            data[k] = out[k][0] + out[k][1]*1j
            print(out[k][0] + out[k][1]*1j)
#         data = []
        
#         for array in out:
             
        
        data_f = np.fft.fft(data)
        freq = np.fft.fftfreq(len(data), d=1/sampling)
        
        data_f = np.fft.fftshift(data_f)
        freq = np.fft.fftshift(freq)
        
#         plt.hist(data)
        print(data_f.shape)
        plt.plot(freq, np.abs(data_f)**2)
        plt.yscale('log')
        plt.title(sampling)
        plt.show()
        
        np.savez(f'{lab_section}-{sampling/1e6}MHz', data = data, location = location, time = time, sample_rate = sampling)
        print(f'Collecting at {sampling/1e6} MHz')
        
        our_sdr.__del__()
        
sample_rates = np.array([1e6, 2e6, 3e6])
lab_section = "6_2_noise"

collection(sample_rates, lab_section)

