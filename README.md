# EXP.NO.5-Simulation-of-Signal-Sampling-Using-Various-Types
5.Simulation of Signal Sampling Using Various Types such as
    i) Ideal Sampling
    ii) Natural Sampling
    iii) Flat Top Sampling

# AIM
To simulate and analyze different types of signal sampling techniques—namely Ideal Sampling, Natural Sampling, and Flat Top Sampling—using suitable software tools, and to understand their characteristics and effects on the original signal.
# SOFTWARE REQUIRED
personal computer 
python
# ALGORITHMS
Natural Sampling Algorithm
Objective: To sample a continuous-time signal using a pulse train, where each pulse's amplitude follows the instantaneous value of the signal.​

Steps:

Define Parameters:

Set the sampling frequency (fs), message signal frequency (fm), and pulse rate.

Create a time vector (t) based on the sampling frequency.​

Generate Message Signal:

Create the message signal (e.g., a sine wave) using the defined frequency and time vector.​

Construct Pulse Train:

Initialize a zero-valued array for the pulse train.

For each period corresponding to the pulse rate, set the pulse amplitude to 1 for a duration determined by the pulse width.​

Perform Natural Sampling:

Multiply the message signal by the pulse train to obtain the naturally sampled signal.​

Reconstruct Sampled Signal:

Extract samples where the pulse train is 1.

Create a time vector for these sampled points.

For visualization, interpolate the sampled signal using zero-order hold or apply a low-pass filter for smoother reconstruction.​

Ideal Sampling Algorithm
Objective: To sample a continuous-time signal using impulse functions, capturing the exact instantaneous values of the signal.​

Steps:

Define Parameters:

Set the sampling frequency (fs) and message signal frequency (fm).

Create a time vector (t) based on the sampling frequency.​
GeeksforGeeks

Generate Message Signal:

Create the message signal (e.g., a sine wave) using the defined frequency and time vector.​

Sample the Signal:

Sample the message signal at the defined sampling frequency to obtain discrete samples.​

Reconstruct the Signal:

Reconstruct the continuous-time signal from the sampled data using interpolation techniques like sinc interpolation.​

Flat Top Sampling Algorithm
Objective: To sample a continuous-time signal using a pulse train with flat-topped pulses, holding the signal value constant during each pulse duration.​
educationallof

Steps:

Define Parameters:

Set the sampling frequency (fs), message signal frequency (fm), and pulse rate.

Create a time vector (t) based on the sampling frequency.​

Generate Message Signal:

Create the message signal (e.g., a sine wave) using the defined frequency and time vector.​

Construct Pulse Train:

Initialize a zero-valued array for the pulse train.

For each period corresponding to the pulse rate, set the pulse amplitude to 1 for a duration determined by the pulse width.​

Perform Flat Top Sampling:

Multiply the message signal by the pulse train to obtain the flat top sampled signal.​

Reconstruct Sampled Signal:

Use the sampled data directly for reconstruction, or apply a low-pass filter to smooth the signal.​


# PROGRAM
NATURAL SAMPLING:


import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter
# Parameters
fs = 1000  # Sampling frequency (samples per second)
T = 1  # Duration in seconds
t = np.arange(0, T, 1/fs)  # Time vector
# Message Signal (sine wave message)
fm = 5  # Frequency of message signal (Hz)
message_signal = np.sin(2 * np.pi * fm * t)
# Pulse Train Parameters
pulse_rate = 50  # pulses per second
pulse_train = np.zeros_like(t)
# Construct Pulse Train (rectangular pulses)
pulse_width = int(fs / pulse_rate / 2)
for i in range(0, len(t), int(fs / pulse_rate)):
  pulse_train[i:i+pulse_width] = 1
# Natural Sampling
nat_signal = message_signal * pulse_train
# Reconstruction (Demodulation) Process
sampled_signal = nat_signal[pulse_train == 1]
# Create a time vector for the sampled points
sample_times = t[pulse_train == 1]
# Interpolation - Zero-Order Hold (just for visualization)
reconstructed_signal = np.zeros_like(t)
for i, time in enumerate(sample_times):
    index = np.argmin(np.abs(t - time))
    reconstructed_signal[index:index+pulse_width] = sampled_signal[i]
# Low-pass Filter (optional, smoother reconstruction)
def lowpass_filter(signal, cutoff, fs, order=5):
    #The following lines were not indented properly, causing the error. 
    #Indenting them fixes the issue.
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return lfilter(b, a, signal)
reconstructed_signal = lowpass_filter(reconstructed_signal,10, fs)
plt.figure(figsize=(14, 10))
# Original Message Signal
plt.subplot(4, 1, 1)
plt.plot(t, message_signal, label='Original Message Signal')
plt.legend()
plt.grid(True)
# Pulse Train
plt.subplot(4, 1, 2)
plt.plot(t, pulse_train, label='Pulse Train')
plt.legend()
plt.grid(True)
# Natural Sampling
plt.subplot(4, 1, 3)
plt.plot(t, nat_signal, label='Natural Sampling')
plt.legend()
plt.grid(True)
# Reconstructed Signal
plt.subplot(4, 1, 4)
plt.plot(t, reconstructed_signal, label='Reconstructed Message Signal', color='green')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

IDEAL SAMPLING:
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import resample
fs = 100
t = np.arange(0, 1, 1/fs) 
f = 5
signal = np.sin(2 * np.pi * f * t)
plt.figure(figsize=(10, 4))
plt.plot(t, signal, label='Continuous Signal')
plt.title('Continuous Signal (fs = 100 Hz)')
plt.xlabel('Time [s]')
plt.ylabel('Amplitude')
plt.grid(True)
plt.legend()
plt.show()
t_sampled = np.arange(0, 1, 1/fs)
signal_sampled = np.sin(2 * np.pi * f * t_sampled)
plt.figure(figsize=(10, 4))
plt.plot(t, signal, label='Continuous Signal', alpha=0.7)
plt.stem(t_sampled, signal_sampled, linefmt='r-', markerfmt='ro', basefmt='r-', label='Sampled Signal (fs = 100 Hz)')
plt.title('Sampling of Continuous Signal (fs = 100 Hz)')
plt.xlabel('Time [s]')
plt.ylabel('Amplitude')
plt.grid(True)
plt.legend()
plt.show()
reconstructed_signal = resample(signal_sampled, len(t))
plt.figure(figsize=(10, 4))
plt.plot(t, signal, label='Continuous Signal', alpha=0.7)
plt.plot(t, reconstructed_signal, 'r--', label='Reconstructed Signal (fs = 100 Hz)')
plt.title('Reconstruction of Sampled Signal (fs = 100 Hz)')
plt.xlabel('Time [s]')
plt.ylabel('Amplitude')
plt.grid(True)
plt.legend()
plt.show()

FLAT TOP SAMPLING:

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter

# Parameters
fs = 1000  # Sampling frequency (samples per second)
T = 1  # Duration in seconds
t = np.arange(0, T, 1/fs)  # Time vector

# Message Signal (sine wave message)
fm = 5  # Frequency of message signal (Hz)
message_signal = np.sin(2 * np.pi * fm * t)

# Pulse Train Parameters
pulse_rate = 50  # pulses per second
pulse_train = np.zeros_like(t)
pulse_width = int(fs / pulse_rate / 2)  # Width of each pulse

# Construct Flat Top Pulse Train
for i in range(0, len(t), int(fs / pulse_rate)):
    pulse_train[i : i + pulse_width] = 1  # Flat top during pulse duration

# Flat Top Sampling
flat_top_signal = message_signal * pulse_train 

# Reconstruction (Demodulation) Process
# In flat-top sampling, reconstruction can be as simple as 
# using the sampled values directly since they represent the message 
# signal during the pulse duration.
reconstructed_signal = flat_top_signal 

# Low-pass Filter (optional, for smoother reconstruction)
def lowpass_filter(signal, cutoff, fs, order=5):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return lfilter(b, a, signal)

reconstructed_signal = lowpass_filter(reconstructed_signal, 10, fs)


# Plotting
plt.figure(figsize=(14, 10))

# Original Message Signal
plt.subplot(4, 1, 1)
plt.plot(t, message_signal, label='Original Message Signal')
plt.legend()
plt.grid(True)

# Pulse Train
plt.subplot(4, 1, 2)
plt.plot(t, pulse_train, label='Flat Top Pulse Train')
plt.legend()
plt.grid(True)

# Flat Top Sampling
plt.subplot(4, 1, 3)
plt.plot(t, flat_top_signal, label='Flat Top Sampling')
plt.legend()
plt.grid(True)

# Reconstructed Signal
plt.subplot(4, 1, 4)
plt.plot(t, reconstructed_signal, label='Reconstructed Message Signal', color='green')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# OUTPUT
 NATURAL SAMPLING:
 ![image](https://github.com/user-attachments/assets/f4d3d948-9a93-4dae-9a00-3869bbf5dd55)
 IDEAL SAMPLING:
 ![image](https://github.com/user-attachments/assets/34a03a87-b9a7-48a9-83c1-24b3dc54bb15)
 FLAT TOP SAMPLING:
 ![Screenshot (41)](https://github.com/user-attachments/assets/a0f15d90-6e09-4626-9d09-66405d78c9ad)

# RESULT / CONCLUSIONS

