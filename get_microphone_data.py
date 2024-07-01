import sys
import sounddevice as sd
import soundfile as sf
import numpy as np
from scipy.signal import butter, filtfilt, spectrogram, find_peaks
import matplotlib.pyplot as plt
import time

from tqdm import tqdm, trange


# Parameters
# TODO - change duration to be dynamic
duration = 10  # Duration of recording in seconds (slightly longer than the actual broadcast)
fs = 44000  # Sampling frequency

bps = 50 # bits per secondes
bit_duration = 1/bps  # Duration of each bit in seconds
cutoff_low = 18000
cutoff_high = 22000
f1 = 21000
f0 = 19000

# TODO!! - change to relevant paramter
amplitude_threshold = 0.1  # Constant amplitude threshold for detecting bits


def fft_cross_correlation(x, y):
    N = len(x) + len(y) - 1
    X = np.fft.fft(x, N)
    Y = np.fft.fft(y, N)
    corr = np.fft.ifft(X * np.conj(Y))
    return np.abs(corr)

def bandpass_filter(data, cutoff_low, cutoff_high, fs):
    # Function to design a bandpass filter
    nyquist = 0.5 * fs
    low = cutoff_low / nyquist
    high = cutoff_high / nyquist
    b, a = butter(N=4, Wn=[low, high], btype='bandpass')
    return filtfilt(b, a, data)

# Function to calculate and plot spectrogram with bit detection lines
def calculate_spectrogram(data, fs, start_idx=0, end_idx=0, bit_positions=0,show=True):
    
    # TODO!! - calculate relevant nprseg and noverlap
    f, t, Sxx = spectrogram(data, fs, nperseg=128, noverlap=64)
    
    # Add a small constant to avoid log of zero
    Sxx += 1e-10
    plt.style.use('dark_background')
    # Plot the spectrogram
    plt.figure(figsize=(10, 5))
    plt.pcolormesh(t, f, 10 * np.log10(Sxx), shading='gouraud')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [s]')
    plt.title(f'Spectrogram ({cutoff_low}-{cutoff_high} Hz)')
    plt.colorbar(label='Intensity [dB]')
    plt.ylim(cutoff_low, cutoff_high)
    
    # Add lines for bit positions
    for bit_time in bit_positions:
        plt.axvline(x=bit_time, color='red', linestyle='--')
    print(start_idx / fs,end_idx / fs)
    plt.axvline(x=start_idx / fs, color='green', linestyle='--', label='Start')
    plt.axvline(x=end_idx / fs, color='blue', linestyle='--', label='End')
    plt.legend()
    if show:
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

def detect_signal(threshold,signal, preamble_bits, fs,bit_duration, high,low,flip = False):
    preamble = encode_data(preamble_bits,bit_duration,fs,high,low)
    preamble_length = len(preamble)
    signal_length = len(signal)
    
    # Ensure the preamble matches the length of the signal segment
    if signal_length > preamble_length:
        preamble_extended = np.pad(preamble, (0, signal_length - preamble_length), mode='constant')
    else:
        preamble_extended = preamble[:signal_length]

    # Cross-correlation using FFT
    corr = fft_cross_correlation(signal,preamble_extended)

    # Use the threshold value from the slider for peak detection
    if flip:
        index = np.argmax(np.flip(corr)>threshold)
    index = np.argmax(corr > threshold)  # Take the first peak
    
    return index - preamble_length/2 ,corr

# Function to create a sine wave for a given frequency
def create_tone(freq, duration, fs):
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)
    tone = 0.5 * np.sin(2 * np.pi * freq * t)
    return tone

# Function to encode data as an audio signal
def encode_data(data, bit_duration, fs, f1, f0):
    signal = np.array([])
    for bit in data:
        if bit == 1:
            bit_tone = create_tone(f1, bit_duration, fs)
        else:
            bit_tone = create_tone(f0, bit_duration, fs)
        signal = np.concatenate((signal, bit_tone))
    return signal

def detect_frequency(segment, fs,f1,f0):
    # Generate reference sine waves for correlation
    t = np.arange(len(segment)) / fs
    ref_wav0 = np.sin(2 * np.pi * f0 * t)
    ref_wav1 = np.sin(2 * np.pi * f1 * t)

    # Normalize the segment and reference signals
    segment_norm = segment / np.linalg.norm(segment)
    ref_wav0_norm = ref_wav0 / np.linalg.norm(ref_wav0)
    ref_wav1_norm = ref_wav1 / np.linalg.norm(ref_wav1)

    # Compute the correlation using FFT
    corr0 = fft_cross_correlation(segment_norm, ref_wav0_norm)
    corr1 = fft_cross_correlation(segment_norm, ref_wav1_norm)

    # Use hypothesis testing to decide the frequency
    if np.max(corr0) > np.max(corr1):
        return "0"
    else:
        return "1"


# Function to decode bits from the audio data
def decode_bits(data, bit_positions, fs, bit_duration,f1,f0):
    num_samples_per_bit = int(bit_duration * fs)
    bits = []

    for start in tqdm(bit_positions):
        segment = data[start : start + num_samples_per_bit]
        if len(segment) == 0:
            continue
        bits.append(detect_frequency(segment,fs,f1,f0))
    return bits


def display_stats(data,filtered_data,corr1,true1,fs):
    # Plot original CSS signal
    plt.subplot(3, 1, 1)
    plt.plot(np.arange(len(data)) / fs, data)
    plt.title('Original CSS Signal')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')

    # Plot filtered signal
    plt.subplot(3, 1, 2)
    plt.plot(np.arange(len(filtered_data)) / fs, filtered_data)
    plt.title('Filtered Signal (18-22 kHz)')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')

    # Plot correlation result
    plt.subplot(3, 1, 3)
    plt.plot(np.arange(len(corr1)) / fs, np.abs(corr1))
    plt.axvline(x=true1/fs, color='r', linestyle='--', label='Average Correlation')
    plt.title('Correlation with Preamble')
    plt.xlabel('Time (s)')
    plt.ylabel('Correlation')
    plt.show()

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
    
    threshold = 5
    total_samples = len(data)
    samples_per_second = total_samples / elapsed_time
    print(f"Total samples recorded: {total_samples}")
    print(f"Samples per second: {samples_per_second:.2f}")
    
    filtered_data = bandpass_filter(data, cutoff_low, cutoff_high, fs)
    start_preamble_bits = [1,0,1,0,1,0,1,0,1,0,1]
    end_preamble_bits = [0,0,0,0]
    start_idx,start_corr = detect_signal(threshold, filtered_data, start_preamble_bits, fs, bit_duration, f1, f0)
    end_idx,end_corr = detect_signal(threshold, filtered_data, end_preamble_bits, fs, bit_duration, f1, f0,True)
    display_stats(data,filtered_data,start_corr,start_idx,fs)
    display_stats(data,filtered_data,end_corr,end_idx,fs)
    bit_positions = np.arange(start_idx/fs, end_idx/fs, bit_duration)
    calculate_spectrogram(filtered_data, fs, start_idx, end_idx,bit_positions)
    bits = decode_bits(filtered_data,bit_positions,fs,bit_duration,f1,f0)
    print(bits)

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









def decode_bfsk_signal(filtered_signal, fs, f0, f1, start_index, bit_duration, threshold_factor=1.5):

    # Step 5: Decode the signal using hypothesis testing
    samples_per_bit = int(bit_duration * fs)
    num_bits = (len(filtered_signal) - start_index) // samples_per_bit

    decoded_bits = []
    for i in range(num_bits):
        start = start_index + i * samples_per_bit
        end = start + samples_per_bit
        segment = filtered_signal[start:end]
        
        corr_f0 = np.sum(np.sin(2 * np.pi * f0 * np.arange(0, bit_duration, 1/fs)) * segment)
        corr_f1 = np.sum(np.sin(2 * np.pi * f1 * np.arange(0, bit_duration, 1/fs)) * segment)
        
        if corr_f1 > corr_f0 * threshold_factor:
            decoded_bits.append(1)
        else:
            decoded_bits.append(0)

    return decoded_bits


