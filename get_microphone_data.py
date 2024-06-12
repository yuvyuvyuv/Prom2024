import sounddevice as sd
import numpy as np
from scipy.signal import butter, filtfilt, spectrogram
import matplotlib.pyplot as plt
import time

# Parameters
duration = 1  # Duration of recording in seconds
fs = 44100  # Sampling frequency
fs = 96000  # Sampling frequency

# Function to design a high-pass filter
def high_pass_filter(data, cutoff_low, cutoff_high, fs):
    nyquist = 0.5 * fs
    low = cutoff_low / nyquist
    high = cutoff_high / nyquist
    b, a = butter(N=4, Wn=[low, high], btype='bandpass')
    y = filtfilt(b, a, data)
    return y

# Function to calculate and plot spectrogram
def calculate_spectrogram(data, fs):
    f, t, Sxx = spectrogram(data, fs, nperseg=1024, noverlap=512)
    
    # Add a small constant to avoid log of zero
    Sxx += 1e-10
    
    # Plot the spectrogram
    plt.pcolormesh(t, f, 10 * np.log10(Sxx), shading='gouraud')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [s]')
    plt.title('Spectrogram (18000-22000 Hz)')
    plt.colorbar(label='Intensity [dB]')
    plt.ylim(18000, 22000)
    plt.show()

# Function to get microphone data
def record_audio(duration, fs):
    print("Recording...")
    start_time = time.time()
    data = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='float64')
    sd.wait()
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Recording complete. Duration: {elapsed_time:.2f} seconds")
    return data.flatten(), elapsed_time

# Main function
def main():
    data, elapsed_time = record_audio(duration, fs)
    
    # Calculate the number of samples received per second
    total_samples = len(data)
    samples_per_second = total_samples / elapsed_time
    print(f"Total samples recorded: {total_samples}")
    print(f"Samples per second: {samples_per_second:.2f}")
    
    # Apply high-pass filter
    filtered_data = high_pass_filter(data, 18000, 22000, fs)
    
    # Calculate and plot spectrogram
    calculate_spectrogram(filtered_data, fs)

if __name__ == "__main__":
    main()
