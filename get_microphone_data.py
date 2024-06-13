import sounddevice as sd
import soundfile as sf
import numpy as np
from scipy.signal import butter, sosfiltfilt, spectrogram, welch
import matplotlib.pyplot as plt
import time
import sys

# Parameters
duration = 6  # Duration of recording in seconds (slightly longer than the actual broadcast)
fs = 96000  # Sampling frequency
bits_per_second = 10
bit_duration = 1/bits_per_second  # Duration of each bit in seconds
cutoff_low = 18000
cutoff_high = 22000

freq1 = 21000  # Frequency for '1' in Hz
freq0 = 19000  # Frequency for '0' in Hz


threshold = 0.01  # Energy threshold for start detection
bit_threshold = 1e-6  # Power threshold for bit detection

# Function to design a bandpass filter
def bandpass_filter(data, cutoff_low, cutoff_high, fs):
    nyquist = 0.5 * fs
    low = cutoff_low / nyquist
    high = cutoff_high / nyquist
    b, a = butter(N=4, Wn=[low, high], btype='bandpass')
    y = sosfiltfilt(b, a, data)
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
    # change energy and data
    # energy = np.convolve(data**2, np.ones(fs) / fs, mode='same')
    start_idx = np.argmax(data > threshold)
    return start_idx


# Function to detect the end of the broadcast using an energy threshold
def detect_end(data, fs, threshold=0.01):
    # change energy and data
    # energy = np.convolve(data**2, np.ones(fs) / fs, mode='same')
    end_idx = len(data) - 1 - np.argmax((data[::-1] > threshold))
    return end_idx

# Function to decode bits from the audio data using a power threshold
def decode_bits_with_threshold(data, fs, bit_duration, bit_threshold):
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
        
        if Pxx[peak_idx] < bit_threshold:
            pass
            # bits.append('0')  # No significant power detected
        elif abs(peak_freq - freq1) < abs(peak_freq - freq0):
            bits.append('1')
        else:
            bits.append('0')
    
    return bits

# Function to refine bit detection
def refine_bit_detection(data, fs, bit_duration, threshold, bit_threshold):
    start_idx = detect_start(data, fs, threshold)
    end_idx = detect_end(data, fs, threshold)
    filtered_data = data[start_idx:end_idx]
    bits = decode_bits_with_threshold(filtered_data, fs, bit_duration, bit_threshold)
    return bits

def main(source, filename=None,fs=fs):
    if source == "live":
        data, elapsed_time = record_audio(duration, fs)
    elif source == "file":
        if filename is None:
            raise ValueError("Filename must be provided when source is 'file'")
        data, fs = sf.read(filename)
        elapsed_time = len(data) / fs
    elif source == "live_save":
        if filename is None:
            raise ValueError("Filename must be provided when source is 'live_save'")
        data, elapsed_time = record_audio(duration, fs)
        sf.write(filename, data, fs)
        print(f"Recording saved to {filename}")
    else:
        raise ValueError("Invalid source. Must be 'live', 'file', or 'live_save'")
        
    # Apply bandpass filter
    filtered_data = bandpass_filter(data, cutoff_low, cutoff_high, fs)
    
    # Refine bit detection
    bits = refine_bit_detection(filtered_data, fs, bit_duration, threshold, bit_threshold)
    print(f"Decoded bits: {''.join(bits)}")
    
    # Calculate and plot spectrogram
    calculate_spectrogram(filtered_data, fs)

if __name__ == "__main__":
   # Parse command line arguments if running from command line
    if len(sys.argv) < 2:
        print("Usage: python your_script.py source [filename]")
        sys.exit(1)
    
    source = sys.argv[1]
    filename = sys.argv[2] if len(sys.argv) > 2 else None
    
    try:
        data, elapsed_time = main(source, filename)
        # Optionally process or analyze the recorded data here
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)