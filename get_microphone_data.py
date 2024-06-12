import sounddevice as sd
import numpy as np
from scipy.signal import butter, filtfilt, spectrogram, find_peaks, welch
import matplotlib.pyplot as plt
import time

# Parameters
duration = 6  # Duration of recording in seconds (slightly longer than the actual broadcast)
fs = 96000  # Sampling frequency
bit_duration = 0.1  # Duration of each bit in seconds
cutoff_low = 18000
cutoff_high = 22000
threshold = 0.01  # Energy threshold for start detection

# Function to design a bandpass filter
def bandpass_filter(data, cutoff_low, cutoff_high, fs):
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
    plt.figure(figsize=(10, 5))
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

# Function to detect the start of the broadcast using an energy threshold
def detect_start(data, fs, threshold=0.01):
    energy = np.convolve(data**2, np.ones(fs) / fs, mode='same')
    start_idx = np.argmax(energy > threshold)
    return start_idx

# Function to decode bits from the audio data
def decode_bits(data, fs, bit_duration):
    num_samples_per_bit = int(bit_duration * fs)
    num_bits = len(data) // num_samples_per_bit
    bits = []
    
    for i in range(num_bits):
        segment = data[i * num_samples_per_bit:(i + 1) * num_samples_per_bit]
        if len(segment) == 0:
            continue
        f, Pxx = welch(segment, fs, nperseg=num_samples_per_bit)
        peak_idx = np.argmax(Pxx)
        peak_freq = f[peak_idx]
        
        if abs(peak_freq - 20000) < abs(peak_freq - 19000):
            bits.append('1')
        else:
            bits.append('0')
    
    return bits

# Function to refine bit detection
def refine_bit_detection(data, fs, bit_duration, threshold):
    start_idx = detect_start(data, fs, threshold)
    filtered_data = data[start_idx:]
    bits = decode_bits(filtered_data, fs, bit_duration)
    return bits

# Main function
def main():
    data, elapsed_time = record_audio(duration, fs)
    
    # Calculate the number of samples received per second
    total_samples = len(data)
    samples_per_second = total_samples / elapsed_time
    print(f"Total samples recorded: {total_samples}")
    print(f"Samples per second: {samples_per_second:.2f}")
    
    # Apply bandpass filter
    filtered_data = bandpass_filter(data, cutoff_low, cutoff_high, fs)
    
    # Refine bit detection
    bits = refine_bit_detection(filtered_data, fs, bit_duration, threshold)
    print(f"Decoded bits: {''.join(bits)}")
    
    # Calculate and plot spectrogram
    calculate_spectrogram(filtered_data, fs)

if __name__ == "__main__":
    main()
