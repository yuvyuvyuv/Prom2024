import sys
import sounddevice as sd
import soundfile as sf
import numpy as np
from scipy.signal import butter, filtfilt, spectrogram
import matplotlib.pyplot as plt
import time

# Parameters
duration = 6  # Duration of recording in seconds (slightly longer than the actual broadcast)
fs = 96000  # Sampling frequency
bit_duration = 0.1  # Duration of each bit in seconds
cutoff_low = 18000
cutoff_high = 22000
amplitude_threshold = 0.9  # Constant amplitude threshold for detecting bits

# Function to design a bandpass filter
def bandpass_filter(data, cutoff_low, cutoff_high, fs):
    nyquist = 0.5 * fs
    low = cutoff_low / nyquist
    high = cutoff_high / nyquist
    b, a = butter(N=4, Wn=[low, high], btype='bandpass')
    y = filtfilt(b, a, data)
    return y

# Function to calculate and plot spectrogram with bit detection lines
def calculate_spectrogram(data, fs, bit_positions, start_idx, end_idx):
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
    
    # Add lines for bit positions
    for bit_time in bit_positions:
        plt.axvline(x=bit_time, color='red', linestyle='--')
    
    plt.axvline(x=start_idx / fs, color='green', linestyle='--', label='Start')
    plt.axvline(x=end_idx / fs, color='blue', linestyle='--', label='End')
    plt.legend()
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

# Function to detect the start and end of the broadcast using amplitude threshold
def detect_start_and_end(data, fs, bit_duration, amplitude_threshold):
    num_samples_per_bit = int(bit_duration * fs)
    start_idx = 0
    end_idx = len(data)

    for i in range(0, len(data) - num_samples_per_bit, num_samples_per_bit):
        segment = data[i:i + num_samples_per_bit]
        fft_segment = np.abs(np.fft.rfft(segment))
        intensity_20000 = fft_segment[int(20000 * len(fft_segment) / (fs / 2))]
        intensity_19000 = fft_segment[int(19000 * len(fft_segment) / (fs / 2))]
        
        if intensity_20000 > amplitude_threshold and intensity_20000 > intensity_19000:
            next_segment = data[i + num_samples_per_bit:i + 2 * num_samples_per_bit]
            next_fft_segment = np.abs(np.fft.rfft(next_segment))
            intensity_19000 = next_fft_segment[int(19000 * len(fft_segment) / (fs / 2))]
            intensity_19000 = next_fft_segment[int(19000 * len(fft_segment) / (fs / 2))]
            
            if intensity_19000 > amplitude_threshold and intensity_20000 < intensity_19000:
                start_idx = i + num_samples_per_bit // 3
                break
    
    for i in range(len(data) - num_samples_per_bit, 0, -num_samples_per_bit):
        segment = data[i - num_samples_per_bit:i]
        fft_segment = np.abs(np.fft.rfft(segment))
        intensity_20000 = fft_segment[int(20000 * len(fft_segment) / (fs / 2))]
        
        if intensity_20000 > amplitude_threshold:
            end_idx = i
            break

    return start_idx, end_idx

# Function to decode bits from the audio data
def decode_bits(data, start_idx, fs, bit_duration):
    num_samples_per_bit = int(bit_duration * fs)
    bits = []
    bit_positions = []

    for i in range(start_idx, len(data) - num_samples_per_bit, num_samples_per_bit):
        segment = data[i:i + num_samples_per_bit]
        if len(segment) == 0:
            continue
        
        fft_segment = np.abs(np.fft.rfft(segment))
        intensity_19000 = fft_segment[int(19000 * len(fft_segment) / (fs / 2))]
        intensity_20000 = fft_segment[int(20000 * len(fft_segment) / (fs / 2))]
        
        bit_positions.append(i / fs + bit_duration)
        
        if intensity_20000 > intensity_19000:
            bits.append('1')
        else:
            bits.append('0')

    return bits, bit_positions

# Main function
def main(source, filename=None, fs=fs):
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
    
    # Calculate the number of samples received per second
    total_samples = len(data)
    samples_per_second = total_samples / elapsed_time
    print(f"Total samples recorded: {total_samples}")
    print(f"Samples per second: {samples_per_second:.2f}")
    
    # Apply bandpass filter
    filtered_data = bandpass_filter(data, cutoff_low, cutoff_high, fs)
    
    # Detect start and end of the broadcast
    start_idx, end_idx = detect_start_and_end(filtered_data, fs, bit_duration, amplitude_threshold)
    print(f"Detected broadcast start at sample index: {start_idx}")
    print(f"Detected broadcast end at sample index: {end_idx}")
    
    # Trim the data 2 bit durations before the start
    start_idx = max(start_idx, 0)
    trimmed_data = filtered_data[start_idx:end_idx]
    
    # Decode bits from the filtered and trimmed data
    bits, bit_positions = decode_bits(trimmed_data, 0, fs, bit_duration)
    print(f"Decoded bits: {''.join(bits)}")
    
    # Calculate and plot spectrogram with bit detection lines
    calculate_spectrogram(trimmed_data, fs, bit_positions, start_idx, end_idx)

if __name__ == "__main__":
    # Parse command line arguments if running from command line
    if len(sys.argv) < 2:
        print("Usage: python your_script.py source [filename]")
        sys.exit(1)

    source = sys.argv[1]
    filename = sys.argv[2] if len(sys.argv) > 2 else None

    try:
        main(source, filename)
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)
